import torch
import wandb

class Checkpointer:
    def __init__(self, output_directory = None, args=None, optimizer=None, scheduler=None, ema=None, model=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.model = model

        self.mAP_max = 0
        self.output_directory = output_directory
        self.args = args

    def restore_if_existing(self, folder, resume_from_best=False):
        checkpoint = self.search_for_checkpoint(folder, best=resume_from_best)
        if checkpoint is not None:
            print(f"Found existing checkpoint at {checkpoint}, resuming...")
            self.restore_checkpoint(folder, best=resume_from_best)

    def mAP_from_checkpoint_name(self, checkpoint_name):
        return float(str(checkpoint_name).split("_")[-1].split(".pth")[0])

    def search_for_checkpoint(self, resume_checkpoint, best=False):
        checkpoints = list(resume_checkpoint.glob("*.pth"))
        if len(checkpoints) == 0:
            return None

        if not best:
            if resume_checkpoint / "last_model.pth" in checkpoints:
                return resume_checkpoint / "last_model.pth"

        # remove "last_model.pth" from checkpoints
        if resume_checkpoint / "last_model.pth" in checkpoints:
            checkpoints.remove(resume_checkpoint / "last_model.pth")

        checkpoints = sorted(checkpoints, key=lambda x: self.mAP_from_checkpoint_name(x.name))
        return checkpoints[-1]


    def restore_if_not_none(self, target, source):
        if target is not None:
            target.load_state_dict(source)

    def restore_checkpoint(self, checkpoint_directory, best=False):
        path = self.search_for_checkpoint(checkpoint_directory, best)
        assert path is not None, "No checkpoint found in {}".format(checkpoint_directory)
        print("Restoring checkpoint from {}".format(path))
        checkpoint = torch.load(path)

        checkpoint['model'] = self.fix_checkpoint(checkpoint['model'])
        checkpoint['ema'] = self.fix_checkpoint(checkpoint['ema'])

        if self.ema is not None:
            self.ema.ema.load_state_dict(checkpoint.get('ema', checkpoint['model']))
            self.ema.updates = checkpoint.get('ema_updates', 0)
        self.restore_if_not_none(self.model, checkpoint['model'])
        self.restore_if_not_none(self.optimizer, checkpoint['optimizer'])
        self.restore_if_not_none(self.scheduler, checkpoint['scheduler'])
        return checkpoint['epoch']

    def fix_checkpoint(self, state_dict):
        return state_dict

    def checkpoint(self, epoch: int, name: str=""):
        self.output_directory.mkdir(exist_ok=True, parents=True)

        checkpoint = {
            "ema": self.ema.ema.state_dict(),
            "ema_updates": self.ema.updates,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "args": self.args
        }

        torch.save(checkpoint, self.output_directory / f"{name}.pth")

    def process(self, data: Dict[str, float], epoch: int):
        mAP = data['mAP']
        data = {f"validation/metric/{k}": v for k, v in data.items()}
        data['epoch'] = epoch
        wandb.log(data)

        if mAP > self.mAP_max:
            self.checkpoint(epoch, name=f"best_model_mAP_{mAP}")
            self.mAP_max = mAP

def smart_load_state_dict(model, weights_path):
    """
    YOLOXの学習済み重みを、自作のDetectorモデルに柔軟にロードする。
    Focus層の任意のチャンネル数（1, 3, 20など）の変更に完全対応。
    """
    checkpoint = torch.load(weights_path, map_location="cpu")
    src_dict = checkpoint.get("model", checkpoint)
    target_dict = model.state_dict()
    
    new_dict = {}
    shape_mismatch = []

    for k, v in src_dict.items():
        clean_k = k
        
        # 1. Main Backboneの接頭辞正規化
        if clean_k.startswith("backbone.backbone."):
            clean_k = clean_k.replace("backbone.backbone.", "backbone.", 1)
        elif any(clean_k.startswith(p) for p in ["dark", "stem"]):
            clean_k = "backbone." + clean_k
            
        # 2. Headの接頭辞正規化
        if clean_k.startswith("head."):
            clean_k = clean_k.replace("head.", "yolox_head.", 1)

        # 3. Main Model (RGB) へのロード
        if clean_k in target_dict:
            if v.shape == target_dict[clean_k].shape:
                new_dict[clean_k] = v
            else:
                shape_mismatch.append(clean_k)

        # 4. Event Backboneへの柔軟なコピー
        if clean_k.startswith("backbone."):
            event_k = clean_k.replace("backbone.", "event_backbone.", 1)
            
            if event_k in target_dict:
                target_shape = target_dict[event_k].shape
                
                # そのまま形が合う場合 (中間層など)
                if v.shape == target_shape:
                    new_dict[event_k] = v
                else:
                    # 【チャンネル数の不一致を吸収する高度なロジック】
                    if len(v.shape) == 4 and len(target_shape) == 4:
                        
                        # パターンA: YOLOXのFocus層 (事前学習が12チャネルで、ターゲットが4の倍数の場合)
                        if v.shape[1] == 12 and target_shape[1] % 4 == 0:
                            c_new = target_shape[1] // 4  # 新しい入力チャンネル数 (例: 1, 20など)
                            
                            # (Out, 12, K, K) -> (Out, 4パッチ, 3チャネル, K, K) に変形
                            v_reshaped = v.view(v.shape[0], 4, 3, v.shape[2], v.shape[3])
                            
                            # 3チャンネルの重みを合計する
                            v_sum = v_reshaped.sum(dim=2, keepdim=True)
                            
                            # 新しいチャンネル数に合わせて複製し、スケールを調整
                            v_adapted = v_sum.repeat(1, 1, c_new, 1, 1) / c_new
                            
                            # 再びFocus層のフラットな形状 (Out, 4 * c_new, K, K) に戻す
                            adapted_v = v_adapted.view(v.shape[0], target_shape[1], v.shape[2], v.shape[3])
                            
                            if adapted_v.shape == target_shape:
                                new_dict[event_k] = adapted_v
                                continue
                                
                        # パターンB: 通常のConv層の最初の層 (RGB 3チャネル -> 任意のCチャネル)
                        elif v.shape[1] == 3:
                            c_new = target_shape[1]
                            v_sum = v.sum(dim=1, keepdim=True)
                            adapted_v = v_sum.repeat(1, c_new, 1, 1) / c_new
                            
                            if adapted_v.shape == target_shape:
                                new_dict[event_k] = adapted_v
                                continue

                    # どうしても形が合わない場合はミスマッチとして記録
                    shape_mismatch.append(event_k)

    # ロード実行
    missing, unexpected = model.load_state_dict(new_dict, strict=False)
    
    # ログ出力 (省略せずに見やすく)
    print("\n" + "="*50)
    print(f"✅ ロード成功: {len(new_dict)} 個のテンソル")
    
    if shape_mismatch:
        print(f"❌ サイズ不一致 (Shape Mismatch): {len(shape_mismatch)} 個")
        for sm in shape_mismatch:
            print(f"   - {sm}")
    print("="*50 + "\n")
        
    return model

def count_parameters(model, model_name="Model"):
    """
    モデルの総パラメータ数、学習可能パラメータ数、固定パラメータ数をカウントして表示する。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\n" + "="*50)
    print(f"📊 {model_name} Parameter Statistics")
    print("-"*50)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Frozen Parameters    : {frozen_params:,}")
    
    # 概算（M: Million）の表示
    print(f"Model Size (approx)  : {total_params / 1e6:.2f} M")
    print("="*50 + "\n")
    
    return total_params, trainable_params