import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import hdf5plugin
import hydra
from omegaconf import OmegaConf, DictConfig

from models.detection.yolox_extension.models.detector import YoloXDetector 
from models.detection.utils.ema import ModelEMA
from data.dsec.dsec_det_dataset import DSECDataset
from data.utils.augmentor import Augmentations
from data.utils.collate import custom_collate_fn
from utils.scheduler import LRSchedule
from utils.helper import Checkpointer
from utils.trainer import Trainer  
from utils.helper import smart_load_state_dict, count_parameters

def seed_everything(seed: int):
    """再現性を確保するためのシード固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):

    output_dir = os.getcwd()

    # 1. シードの固定
    seed = config.get("seed", 42)
    seed_everything(seed)

    # 2. WandBの初期化 
    wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        config=OmegaConf.to_container(config, resolve=True), 
        dir=output_dir  
    )

    print("=== 🚀 Initializing Datasets ===")
    augmentations = Augmentations(config=config.dataset.transforms)
    
    train_dataset = DSECDataset(
        root=Path(config.dataset.dataset_root), 
        split="train", 
        sync=config.dataset.sync,
        transforms=augmentations.transform_training,
        split_config=config.dataset.split,
        use_image=config.dataset.use_image,
        use_events=config.dataset.use_events,
    )
    val_dataset = DSECDataset(
        root=Path(config.dataset.dataset_root), 
        split="val", 
        sync=config.dataset.sync,
        transforms=augmentations.transform_testing,
        split_config=config.dataset.split,
        use_image=config.dataset.use_image,
        use_events=config.dataset.use_events,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        num_workers=config.training.num_workers, 
        drop_last=True,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.training.num_workers, 
        drop_last=False,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS (Apple Silicon) 用
    else:
        device = torch.device("cpu")  
    
    print(f"=== 🚀 Using Device: {device} ===")

    print("=== 🚀 Initializing Model ===")
    model = YoloXDetector(config.model).to(device)
    model = smart_load_state_dict(model, config.training.pretrained_path)
    count_parameters(model, model_name="YoloXDetector")
    ema = ModelEMA(model, decay=config.training.get("ema_decay", 0.9999))

    print("=== 🚀 Initializing Optimizer & Scheduler ===")
    nominal_batch_size = 64
    scaled_lr = config.training.lr * np.sqrt(config.training.batch_size) / np.sqrt(nominal_batch_size)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=scaled_lr, 
        weight_decay=config.training.weight_decay
    )

    # Schedulerの設定
    num_iters_per_epoch = len(train_loader)
    lr_func = LRSchedule(
        warmup_epochs=config.training.warmup_epochs,
        num_iters_per_epoch=num_iters_per_epoch,
        tot_num_epochs=config.training.tot_num_epochs
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    print("=== 🚀 Initializing Checkpointer & Trainer ===")
    checkpointer = Checkpointer(
        output_directory=config.training.ckpt_dir,
        model=model, 
        optimizer=optimizer,
        scheduler=lr_scheduler, 
        ema=ema,
        args=config 
    )

    # Trainerの初期化 (先ほど作成した完全版)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpointer=checkpointer,
        scheduler=lr_scheduler,
        ema=ema,
        device=device,
        config=config 
    )

    # 学習・評価ループの開始
    trainer.run()
    
    # 終了処理
    wandb.finish()


if __name__ == "__main__":
    main()