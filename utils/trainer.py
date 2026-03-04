import torch
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
        
        training_config = self.config.get('training', {})
        self.use_amp = training_config.get('use_amp', True)
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        
    def _fix_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0)

    def _to_buffer_dict(self, boxes, labels, scores=None):
        """DAGRの設計に倣い、Bufferが期待する最小構成の辞書を生成する"""
        res = {
            "boxes": boxes.detach().cpu(),
            "labels": labels.detach().cpu().long()
        }
        if scores is not None:
            res["scores"] = scores.detach().cpu()
        return res

    def _prepare_eval_batch(self, outputs, targets):
        """
        1バッチ分の出力を一括で整形。
        GTのパディング除去と座標変換、予測値のスコア計算をここで行う。
        """
        formatted_outputs = []
        formatted_targets = []

        # --- 1. 推論結果の整形 ---
        # outputsはpostprocess後のList[Tensor or None]を想定
        for out in outputs:
            if out is None or len(out) == 0:
                formatted_outputs.append(self._to_buffer_dict(
                    torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,))
                ))
            else:
                # out: [x1, y1, x2, y2, obj, cls_conf, label]
                boxes = out[:, :4]
                scores = out[:, 4] * out[:, 5]
                labels = out[:, 6]
                formatted_outputs.append(self._to_buffer_dict(boxes, labels, scores))

        # --- 2. 正解データ(GT)の整形 ---
        if targets is not None:
            # targets: (B, Max_Obj, 5) -> [class_id, cx, cy, w, h]
            for i in range(targets.size(0)):
                target = targets[i]
                mask = (target[:, 3] > 0) & (target[:, 4] > 0) # w,h > 0
                valid_t = target[mask]
                
                # 🌟 [cx, cy, w, h] -> [x1, y1, x2, y2] に正しく変換する
                boxes = valid_t[:, 1:5].clone()
                boxes_xyxy = torch.zeros_like(boxes)
                
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # x1 = cx - w/2
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # y1 = cy - h/2
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0  # x2 = cx + w/2
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0  # y2 = cy + h/2
                
                formatted_targets.append(self._to_buffer_dict(boxes_xyxy, valid_t[:, 0]))

        return formatted_outputs, formatted_targets

    def _run_eval_step(self, data, model, mapcalc):
        """評価の1ステップ(Forward + PostProcess + BufferUpdate)を集約"""
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            # retrieve_detections=True はモデル側の実装に依存
            outputs, _ = model(**data, retrieve_detections=True)
        
        if outputs is not None:
            # 後処理
            pred_processed = postprocess(
                prediction=outputs,
                num_classes=self.config.model.head.num_classes, 
                conf_thre=self.config.model.postprocess.confidence_threshold,
                nms_thre=self.config.model.postprocess.nms_threshold
            )
            
            # 整形とバッファへの追加
            f_outs, f_tgts = self._prepare_eval_batch(pred_processed, data.get('targets'))
            mapcalc.update(
                f_outs, f_tgts, 
                dataset=self.val_loader.dataset.dataset_name, 
                height=self.val_loader.dataset.height, 
                width=self.val_loader.dataset.width
            )

    def _sanity_check(self, dry_run_steps=2):
        print("=== 🚀 Running Sanity Check ===")
        try:
            # Training Test
            self.model.train()
            train_iter = iter(self.train_loader)
            for _ in range(dry_run_steps):
                data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in next(train_iter).items()}
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    _, losses = self.model(**data)
            print("✅ Training Forward: OK")

            # Evaluation Test
            eval_model = self.ema.ema if self.ema is not None else self.model
            eval_model.eval()
            val_iter = iter(self.val_loader)
            mapcalc = DetectionBuffer(self.val_loader.dataset.height, self.val_loader.dataset.width, self.val_loader.dataset.classes)
            
            with torch.no_grad():
                for _ in range(dry_run_steps):
                    data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in next(val_iter).items()}
                    self._run_eval_step(data, eval_model, mapcalc)
            
            mapcalc.compute()
            print("✅ Evaluation Compute: OK\n")
        except Exception as e:
            print(f"❌ Sanity Check Failed: {e}")
            raise e

    def train_epoch(self, epoch):
        self.model.train()
        total_epoch_loss = 0.0
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        
        for data in pbar:
            data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in data.items()}
            
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                _, losses = self.model(**data)
                total_loss = losses.get('loss', losses.get('total_loss'))

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if hasattr(self.config.training, 'clip'):
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.training.clip)
            self._fix_gradients()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None: self.scheduler.step()
            if self.ema is not None: self.ema.update(self.model)

            total_epoch_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})
            if wandb.run is not None:
                wandb.log({f"train/{k}": (v.item() if isinstance(v, torch.Tensor) else v) for k, v in losses.items()})

                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[-1]
                    wandb.log({"train/lr": current_lr})


        return total_epoch_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, epoch):
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        mapcalc = DetectionBuffer(self.val_loader.dataset.height, self.val_loader.dataset.width, self.val_loader.dataset.classes)
        
        pbar = tqdm.tqdm(self.val_loader, desc=f"Epoch {epoch} Evaluating")
        for data in pbar:
            data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            self._run_eval_step(data, eval_model, mapcalc)

        metrics = mapcalc.compute()
        if wandb.run is not None:
            wandb.log({f"val/{k}": v for k, v in metrics.items()})
        
        return metrics

    def run(self):
        # レジューム処理
        self.start_epoch = self.checkpointer.restore_if_existing(best=False)
        self._sanity_check()

        for epoch in range(self.start_epoch, self.config.training.tot_num_epochs):
            train_loss = self.train_epoch(epoch)
            if epoch % self.config.training.eval_interval == 0:
                metrics = self.evaluate(epoch)
                self.checkpointer.process(metrics, epoch)
                self.checkpointer.checkpoint(epoch, name="last_model")