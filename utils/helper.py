import torch
from pathlib import Path
from typing import Dict

class Checkpointer:
    def __init__(self, output_directory, args=None, optimizer=None, scheduler=None, ema=None, model=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.model = model

        self.mAP_max = 0
        self.output_directory = Path(output_directory) 
        self.args = args

    def mAP_from_checkpoint_name(self, checkpoint_name):
        try:
            return float(str(checkpoint_name).split("_")[-1].split(".pth")[0])
        except ValueError:
            return 0.0

    def search_for_checkpoint(self, directory: Path, best=False):
        """ディレクトリ内のチェックポイントを探す"""
        if not directory.exists():
            return None

        checkpoints = list(directory.glob("*.pth"))
        if len(checkpoints) == 0:
            return None

        # ベストモデルを探す場合
        if best:
            # mAPが含まれるファイル名のみ抽出
            best_checkpoints = [c for c in checkpoints if "best_model_mAP" in c.name]
            if not best_checkpoints:
                return None
            # mAPの値でソートして一番高いものを返す
            best_checkpoints = sorted(best_checkpoints, key=lambda x: self.mAP_from_checkpoint_name(x.name))
            return best_checkpoints[-1]

        # 最新のモデルを探す場合
        last_model_path = directory / "last_model.pth"
        if last_model_path in checkpoints:
            return last_model_path
        
        return None

    def restore_if_existing(self, best=False):
        """出力ディレクトリにチェックポイントがあればレジュームする"""
        checkpoint_path = self.search_for_checkpoint(self.output_directory, best=best)
        if checkpoint_path is not None:
            print(f"Found existing checkpoint at {checkpoint_path}, resuming...")
            return self.restore_checkpoint(checkpoint_path)
        return 0 # 見つからなければエポック0からスタート

    def restore_checkpoint(self, checkpoint_path):
        """指定されたパスのチェックポイントを読み込む"""
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f"No checkpoint found at {checkpoint_path}"
        
        print(f"Restoring checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # モデルの重みを復元
        if self.model is not None and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            
        # EMAの重みを復元
        if self.ema is not None:
            ema_state = checkpoint.get('ema', checkpoint.get('model'))
            if ema_state is not None:
                self.ema.ema.load_state_dict(ema_state)
            self.ema.updates = checkpoint.get('ema_updates', 0)
            
        # OptimizerとSchedulerの復元
        if self.optimizer is not None and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        return checkpoint.get('epoch', 0) + 1 

    def checkpoint(self, epoch: int, name: str=""):
        self.output_directory.mkdir(exist_ok=True, parents=True)

        checkpoint = {
            "epoch": epoch,
            "args": self.args
        }
        
        if self.model is not None:
            checkpoint["model"] = self.model.state_dict()
        if self.ema is not None:
            checkpoint["ema"] = self.ema.ema.state_dict()
            checkpoint["ema_updates"] = self.ema.updates
        if self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.output_directory / f"{name}.pth")

    def process(self, metrics: Dict[str, float], epoch: int):
        """評価指標を受け取り、WandBへの記録とベストモデルの保存を行う"""
        mAP = metrics.get('mAP', 0.0) 

        if mAP > self.mAP_max:
            print(f"🌟 New Best mAP: {mAP:.4f} (Previous: {self.mAP_max:.4f}). Saving model...")
            self.checkpoint(epoch, name=f"best_model_mAP_{mAP:.4f}")
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