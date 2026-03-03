import os
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from models.detection.yolox_extension.models.detector import YoloXDetector 
from data.dsec.dsec_det_dataset import DSECDataset
from data.utils.augmentor import Augmentations
from data.utils.collate import custom_collate_fn
from utils.trainer import Trainer  
from utils.helper import smart_load_state_dict

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
    model = YoloXDetector(config.model).to(device)
    
    # テスト用の重みをロード (例: config.evaluation.ckpt_path)
    ckpt_path = config.ckpt_path
    model = smart_load_state_dict(model, ckpt_path)

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

    output_path = os.path.join(os.getcwd(), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Results saved to {output_path}")

if __name__ == "__main__":
    main()