import torch
import torchvision
import tqdm
import wandb
import numpy as np

from utils.buffers import DetectionBuffer
from models.detection.yolox.utils.boxes import postprocess

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, checkpointer, 
                 scheduler=None, ema=None, device='cuda', config=None):
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpointer = checkpointer 
        self.scheduler = scheduler
        self.ema = ema
        self.config = config
        self.start_epoch = 0
        
        # AMP設定
        self.use_amp = getattr(self.config.training, 'use_amp', True)
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        
    def _fix_gradients(self):
        """勾配の爆発やNaNを防ぐ安全策"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0)

    def _format_targets_for_eval(self, targets):
        """バッチ化された正解テンソルをDetectionBufferが期待する辞書のリストに変換する"""
        formatted = []
        if targets is None:
            return formatted
            
        # targets shape: (Batch, Max_Objects, 5) => [class_id, x, y, w, h]
        for i in range(targets.size(0)):
            target = targets[i]
            
            # パディング(ゼロ埋め)を除外：幅(w)と高さ(h)が0より大きいものを有効とする
            valid_mask = (target[:, 3] > 0) & (target[:, 4] > 0)
            valid_target = target[valid_mask]
            
            labels = valid_target[:, 0].long()
            
            # [x, y, w, h] を [x1, y1, x2, y2] に変換して評価バッファに渡す
            boxes = valid_target[:, 1:5].clone()
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
            
            formatted.append({
                "boxes": boxes,
                "labels": labels
            })
        return formatted
    
    def _format_outputs_for_eval(self, outputs):
        """YOLOXのテンソル出力を、DetectionBufferが期待する辞書のリストに変換する"""
        formatted = []
        for out in outputs:
            if out is None or len(out) == 0:
                # 検出ゼロの場合
                formatted.append({
                    "boxes": torch.zeros((0, 4), device=self.device),
                    "scores": torch.zeros((0,), device=self.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=self.device)
                })
            elif isinstance(out, torch.Tensor):
                # postprocess後の出力: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
                boxes = out[:, :4]
                # スコアは objectness * class_confidence
                scores = out[:, 4] * out[:, 5]
                labels = out[:, 6].long()
                
                formatted.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                })
            else:
                # すでに辞書型ならそのまま
                formatted.append(out)
        return formatted

    def _sanity_check(self, dry_run_steps=2):
        """学習開始前にForward/Loss計算が正しく動くか数ステップだけテストする"""
        print("=== 🚀 Running Sanity Check (Forward Test) ===")
        
        try:
            # 1. Trainingのテスト (ForwardとLoss計算が通るか)
            self.model.train()
            train_iter = iter(self.train_loader)
            for _ in range(dry_run_steps):
                data = next(train_iter)
                if isinstance(data, dict):
                    data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                            for k, v in data.items()}
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                        outputs, losses = self.model(**data)
                else:
                    raise ValueError("Trainer expects 'data' to be a dictionary.")
            print("✅ Training forward pass (AMP): OK")

            # 2. Evaluationのテスト (EMAモデルでのForwardが通るか)
            eval_model = self.ema.ema if self.ema is not None else self.model
            eval_model.eval()
            val_iter = iter(self.val_loader)
            
            mapcalc = DetectionBuffer(
                height=self.val_loader.dataset.height, 
                width=self.val_loader.dataset.width, 
                classes=self.val_loader.dataset.classes
            )
            
            with torch.no_grad():
                for _ in range(dry_run_steps):
                    data = next(val_iter)
                    if isinstance(data, dict):
                        data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in data.items()}
                        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                            outputs, _ = eval_model(**data, retrieve_detections=True)
                        
                        if outputs is not None:
                            # 🌟 後処理 (NMSと座標変換)
                            # YAMLの構造に合わせて config.model.head.num_classes 等に適宜変更してください
                            pred_processed = postprocess(
                                prediction=outputs,
                                num_classes=self.config.model.head.num_classes, 
                                conf_thre=self.config.model.postprocess.confidence_threshold,
                                nms_thre=self.config.model.postprocess.nms_threshold
                            )

                            formatted_outputs = self._format_outputs_for_eval(pred_processed)
                            
                            targets = data.get('targets')
                            formatted_targets = self._format_targets_for_eval(targets)
                            
                            mapcalc.update(
                                formatted_outputs,
                                formatted_targets, 
                                dataset=self.val_loader.dataset.dataset_name, 
                                height=self.val_loader.dataset.height, 
                                width=self.val_loader.dataset.width
                            )
                    else:
                        raise ValueError("Trainer expects 'data' to be a dictionary.")
            
            mapcalc.compute()
            print("✅ Evaluation forward pass & metrics compute: OK")
            print("============================================\n")
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()
            
        except StopIteration:
            print("⚠️ データセットのサイズがdry_run_stepsより小さいため中断されましたが、問題ありません。")
        except Exception as e:
            print("❌ Sanity Check Failed! 学習開始前にエラーが発生しました。")
            raise e

    def train_epoch(self, epoch):
        self.model.train()
        total_epoch_loss = 0.0
        
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        for batch_idx, data in enumerate(pbar):
            if isinstance(data, dict):
                data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in data.items()}
            else:
                raise ValueError("Trainer expects 'data' to be a dictionary matching model arguments.")

            self.optimizer.zero_grad(set_to_none=True)

            # --- 1. 順伝播とLoss計算をAMPコンテキストでラップ ---
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                outputs, losses = self.model(**data)

                # YOLOXの辞書からメインのロスを取得
                total_loss = losses.get('loss', losses.get('total_loss'))
                if total_loss is None:
                    raise KeyError("Loss dictionary must contain 'loss' or 'total_loss' for backward pass.")

                # WandB用ログの作成: 辞書の全要素をloopしてログ用に整形
                loss_logs = {
                    f"train/{k}": (v.item() if isinstance(v, torch.Tensor) else float(v)) 
                    for k, v in losses.items()
                }

            # --- 2. スケーリングされたLossで逆伝播 ---
            self.scaler.scale(total_loss).backward()

            # --- 3. 勾配のスケールを戻す (Clippingや手動操作の前に必須) ---
            self.scaler.unscale_(self.optimizer)

            if hasattr(self.config.training, 'clip'):
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.training.clip)
            self._fix_gradients()

            # --- 4. OptimizerのステップとScalerの更新 ---
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            total_epoch_loss += total_loss.item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # ログの記録（ステップ単位）
            wandb.log({"train/lr": current_lr, **loss_logs})
            pbar.set_postfix({'loss': total_loss.item(), 'lr': current_lr})

        return total_epoch_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, epoch):
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        
        mapcalc = DetectionBuffer(
            height=self.val_loader.dataset.height, 
            width=self.val_loader.dataset.width, 
            classes=self.val_loader.dataset.classes
        )
        
        pbar = tqdm.tqdm(self.val_loader, desc=f"Epoch {epoch} Evaluating")
        for batch_idx, data in enumerate(pbar):
            if isinstance(data, dict):
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in data.items()}
            else:
                raise ValueError("Evaluation expects 'data' to be a dictionary.")

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                outputs, _ = eval_model(**data, retrieve_detections=True)
            
            if outputs is not None:
                # 🌟 後処理 (NMSと座標変換)
                pred_processed = postprocess(
                    prediction=outputs,
                    num_classes=self.config.model.head.num_classes, 
                    conf_thre=self.config.model.postprocess.confidence_threshold,
                    nms_thre=self.config.model.postprocess.nms_threshold
                )
                         
                formatted_outputs = self._format_outputs_for_eval(pred_processed)
                
                targets = data.get('targets')
                formatted_targets = self._format_targets_for_eval(targets) 
                
                mapcalc.update(
                    formatted_outputs, 
                    formatted_targets, 
                    dataset=self.val_loader.dataset.dataset_name, 
                    height=self.val_loader.dataset.height, 
                    width=self.val_loader.dataset.width
                )
            
            if batch_idx % 10 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                pbar.set_postfix({'current_batch': batch_idx})

        metrics = mapcalc.compute()
        wandb.log({f"val/{k}": v for k, v in metrics.items()})
        return metrics

    def run(self):
        # 1. レジューム処理
        resume_path = getattr(self.config.training, 'resume_checkpoint', None)
        if resume_path:
            self.start_epoch = self.checkpointer.restore_checkpoint(resume_path)
        else:
            self.start_epoch = self.checkpointer.restore_if_existing(best=False)

        # 2. Sanity Check (学習開始前にテストを実行)
        self._sanity_check(dry_run_steps=2)

        # 3. 学習・評価ループ
        print("=== 🚀 Starting Training Loop ===")
        tot_num_epochs = getattr(self.config.training, 'tot_num_epochs', 100)
        eval_interval = getattr(self.config.training, 'eval_interval', 1)

        for epoch in range(self.start_epoch, tot_num_epochs):
            train_loss = self.train_epoch(epoch)
            wandb.log({"train/epoch_loss": train_loss, "epoch": epoch})
            
            if epoch % eval_interval == 0:
                metrics = self.evaluate(epoch)
                
                self.checkpointer.process(metrics, epoch)
                self.checkpointer.checkpoint(epoch, name="last_model")