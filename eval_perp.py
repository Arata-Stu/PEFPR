import os
import shutil 
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from models.detection.yolox_extension.models.detector import YoloXPEPRDetector 
from data.dsec.dsec_det_dataset import DSECDataset
from data.utils.augmentor import Augmentations
from data.utils.collate import custom_collate_fn
from utils.trainer import Trainer  
from utils.helper import count_parameters

@hydra.main(config_path='config', config_name='eval', version_base='1.2')
def main(config: DictConfig):
    
    print("=== 🚀 Initializing Test Dataset ===")
    augmentations = Augmentations(config=config.dataset.transforms)
    
    test_dataset = DSECDataset(
        root=Path(config.dataset.dataset_root), 
        split="test", 
        sync=config.dataset.sync,
        transforms=augmentations.transform_testing,
        split_config=config.dataset.split,
        use_image=config.dataset.use_image,
        use_events=config.dataset.use_events,
        event_repr_config=config.dataset.get("event_repr", None)
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.eval.batch_size, 
        shuffle=False, 
        num_workers=config.eval.num_workers, 
        drop_last=False,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 🚀 Using Device: {device} ===")

    print("=== 🚀 Initializing Model & Loading Weights ===")
    model = YoloXPEPRDetector(config.model).to(device)
    count_parameters(model, model_name="PERPYoloXDetector")
    
    ckpt_path = config.ckpt_path
    print(f"Loading checkpoint from {ckpt_path}")
    
    # 1. PyTorch 2.6対策をして標準ロード
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 2. EMAの重みを最優先で抽出
    if "ema" in checkpoint:
        print("✅ EMA weights found. Loading EMA state_dict...")
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        print("⚠️ EMA weights NOT found. Loading standard model state_dict...")
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # 3. 厳密なロード (strict=True) でキーの完全一致を保証
    model.load_state_dict(state_dict, strict=True)
    print("✅ Weights loaded successfully!")

    print("=== 🚀 Starting Evaluation ===")
    evaluator = Trainer(
        model=model,
        optimizer=None,      
        train_loader=None,   
        val_loader=test_loader, 
        checkpointer=None,   
        scheduler=None,      
        ema=None,            
        device=device,
        config=config 
    )

    # 評価を実行
    metrics = evaluator.evaluate(epoch="test")
    
    print("\n" + "="*40)
    print("🎯 Evaluation Results")
    print("="*40)
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
    print("="*40)

    # --- 保存処理の統合 ---
    output_dir = config.eval.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 評価結果の保存
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Results saved to {results_path}")

    # 2. Configの保存 (OmegaConfを使ってYAML形式で保存)
    config_path = os.path.join(output_dir, "eval_config.yaml")
    OmegaConf.save(config=config, f=config_path)
    print(f"✅ Config saved to {config_path}")

    # 3. 評価に使用した重みファイルのコピー
    ckpt_dst_path = os.path.join(output_dir, os.path.basename(ckpt_path))
    try:
        shutil.copy2(ckpt_path, ckpt_dst_path)
        print(f"✅ Checkpoint copied to {ckpt_dst_path}")
    except Exception as e:
        print(f"⚠️ Failed to copy checkpoint: {e}")

    print("\n" + "="*40)
    print(f"🎉 All evaluation artifacts are saved in:\n📁 {output_dir}")
    print("="*40)

if __name__ == "__main__":
    main()