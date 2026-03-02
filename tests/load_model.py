import sys
sys.path.append("../")
import torch
from omegaconf import OmegaConf

from models.detection.yolox_extension.models.detector import YoloXDetector, YoloXPEPRDetector
from utils.helper import smart_load_state_dict, count_parameters

# 1. --- モデルの準備 ---
yaml_path = "../config/yolox.yaml"
model_config = OmegaConf.load(yaml_path)
model = YoloXDetector(model_config)

# 2. --- 重みのロード ---
weights_path = "../weights/yolox/yolox_l.pth"
model = smart_load_state_dict(model, weights_path)
count_parameters(model, model_name="YoloXDetector")
# 3. --- 推論テスト (Dummy Input) ---
model.eval() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

img_size = model_config.get("input_size", 640)
dummy_img = torch.randn(1, 3, img_size, img_size).to(device)

try:
    with torch.no_grad():
        backbone_features = model.forward_backbone(dummy_img)
        outputs, losses = model.forward_detect(backbone_features)
        
    print("✅ 推論に成功しました！")
    
    if isinstance(backbone_features, dict):
        for stage, feat in backbone_features.items():
            print(f"   Stage {stage} backbone feature shape: {list(feat.shape)}")
    else:
        print(f"   Backbone feature shape: {list(backbone_features.shape)}")

    if isinstance(outputs, dict):
        for stage, feat in outputs.items():
            print(f"   Stage {stage} output shape: {list(feat.shape)}")
    else:
        print(f"   Output shape: {list(outputs.shape)}")

except Exception as e:
    print(f"❌ 推論中にエラーが発生しました:\n{e}")

print("\nすべての処理が完了しました。")


# PERP

# 1. --- モデルの準備 ---
yaml_path = "../config/PEPR-yolox.yaml"
model_config = OmegaConf.load(yaml_path)
model = YoloXPEPRDetector(model_config)

# 2. --- 重みのロード ---
weights_path = "../weights/yolox/yolox_l.pth"
model = smart_load_state_dict(model, weights_path)
count_parameters(model, model_name="YoloXPEPRDetector")
# 3. --- 推論テスト (Dummy Input) ---
model.eval() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

img_size = model_config.get("input_size", 640)
dummy_img = torch.randn(1, 3, img_size, img_size).to(device)

try:
    with torch.no_grad():
        backbone_features = model.forward_backbone(dummy_img)
        outputs, losses = model.forward_detect(backbone_features)
        
    print("✅ 推論に成功しました！")
    
    if isinstance(backbone_features, dict):
        for stage, feat in backbone_features.items():
            print(f"   Stage {stage} backbone feature shape: {list(feat.shape)}")
    else:
        print(f"   Backbone feature shape: {list(backbone_features.shape)}")

    if isinstance(outputs, dict):
        for stage, feat in outputs.items():
            print(f"   Stage {stage} output shape: {list(feat.shape)}")
    else:
        print(f"   Output shape: {list(outputs.shape)}")

except Exception as e:
    print(f"❌ 推論中にエラーが発生しました:\n{e}")

print("\n--- 学習モード (Training) のテストを開始します ---")
model.train()

# 1. バウンディングボックス座標 (x, y, w, h) を 0 ~ img_size の範囲で生成
dummy_bboxes = torch.rand(1, 50, 4) * img_size

# 2. クラスID を 0 ~ 3 (num_classes=4の場合) の範囲の整数として生成し、float型に合わせる
dummy_classes = torch.randint(0, 3, (1, 50, 1)).float()

# 3. 結合して (1, 50, 5) のターゲットテンソルを作成 [class_id, x, y, w, h] の並びを想定
dummy_targets = torch.cat([dummy_classes, dummy_bboxes], dim=-1).to(device)

# ダミーのイベントデータ (1チャネル)
dummy_event = torch.randn(1, 1, img_size, img_size).to(device)


outputs, losses = model(
        x=dummy_img, 
        event_data=dummy_event, 
        targets=dummy_targets
    )
for k, v in losses.items():
        print(f"      - {k}: {v.item():.4f}" if isinstance(v, torch.Tensor) else f"      - {k}: {v}")
        
print("\nすべての処理が完了しました。")
