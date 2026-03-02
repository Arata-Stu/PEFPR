import torch
import tqdm
import wandb

from utils.buffers import DetectionBuffer

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, checkpointer, 
                 scheduler=None, ema=None, device='cuda', args=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpointer = checkpointer 
        self.scheduler = scheduler
        self.ema = ema
        self.device = device
        self.args = args
        self.start_epoch = 0
        self.use_amp = getattr(self.args, 'use_amp', True)
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        
    def _fix_gradients(self):
        """勾配の爆発やNaNを防ぐ安全策"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0)

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
                    # AMPコンテキストでのテスト
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                        outputs, losses = self.model(**data)
                else:
                    raise ValueError("Trainer expects 'data' to be a dictionary.")
            print("✅ Training forward pass (AMP): OK")

            # 2. Evaluationのテスト (EMAモデルでのForwardが通るか)
            eval_model = self.ema.ema if self.ema is not None else self.model
            eval_model.eval()
            val_iter = iter(self.val_loader)
            
            # 評価用バッファの初期化テスト
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
                        # 評価時のAMPテスト
                        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                            outputs, _ = eval_model(**data, retrieve_detections=True)
                        
                        if outputs is not None:
                            targets = data.get('targets')
                            mapcalc.update(
                                outputs, 
                                targets, 
                                dataset=self.val_loader.dataset.dataset_name, 
                                height=self.val_loader.dataset.height, 
                                width=self.val_loader.dataset.width
                            )
                    else:
                        raise ValueError("Trainer expects 'data' to be a dictionary.")
            
            # 評価指標計算のテスト
            mapcalc.compute()
            print("✅ Evaluation forward pass & metrics compute: OK")
            print("============================================\n")
            
            # キャッシュをクリアして本番に備える
            torch.cuda.empty_cache()
            
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

                if isinstance(losses, dict):
                    total_loss = sum(loss for loss in losses.values())
                    loss_logs = {f"train/{k}": v.item() for k, v in losses.items()}
                elif isinstance(losses, torch.Tensor):
                    total_loss = losses
                    loss_logs = {"train/loss": total_loss.item()}
                else:
                    raise ValueError("Model must return 'losses' as a dict of Tensors or a single Tensor.")

            # --- 2. スケーリングされたLossで逆伝播 ---
            self.scaler.scale(total_loss).backward()

            # --- 3. 勾配のスケールを戻す (Clippingや手動操作の前に必須) ---
            self.scaler.unscale_(self.optimizer)

            if hasattr(self.args, 'clip'):
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip)
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
            wandb.log({"train/step_total_loss": total_loss.item(), "train/lr": current_lr, **loss_logs})
            pbar.set_postfix({'total_loss': total_loss.item(), 'lr': current_lr})

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

            # --- 推論時もAMPを適用 ---
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                outputs, _ = eval_model(**data, retrieve_detections=True)
            
            if outputs is not None:
                targets = data.get('targets')
                mapcalc.update(
                    outputs, 
                    targets, 
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
        resume_path = getattr(self.args, 'resume_checkpoint', None)
        if resume_path:
            # 明示的にパスが指定された場合
            self.start_epoch = self.checkpointer.restore_checkpoint(resume_path)
        else:
            # 出力先ディレクトリから自動的にレジューム
            self.start_epoch = self.checkpointer.restore_if_existing(best=False)

        # 2. Sanity Check (学習開始前にテストを実行)
        self._sanity_check(dry_run_steps=2)

        # 3. 学習・評価ループ
        print("=== 🚀 Starting Training Loop ===")
        for epoch in range(self.start_epoch, self.args.tot_num_epochs):
            train_loss = self.train_epoch(epoch)
            wandb.log({"train/epoch_loss": train_loss, "epoch": epoch})
            
            eval_interval = getattr(self.args, 'eval_interval', 1)
            if epoch % eval_interval == 0:
                metrics = self.evaluate(epoch)
                
                self.checkpointer.process(metrics, epoch)
                self.checkpointer.checkpoint(epoch, name="last_model")