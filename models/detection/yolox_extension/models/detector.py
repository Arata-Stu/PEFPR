from typing import Dict, Optional, Tuple, Union

import torch as th
import torch.nn.functional as F
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from .build import build_yolox_fpn, build_yolox_head, build_yolo_backbone, build_pepr_predictor
from utils.timers import TimerDummy as CudaTimer


class YoloXDetector(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = build_yolo_backbone(backbone_cfg)

        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.yolox_head = build_yolox_head(head_cfg, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor,):
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features = self.backbone(x)
        return backbone_features

    def forward_detect(self,
                       backbone_features,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.yolox_head(fpn_features, targets)
            return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.yolox_head(fpn_features)
        assert losses is None
        return outputs, losses

    def forward(self,
                x: th.Tensor,
                retrieve_detections: bool = True,
                targets: Optional[th.Tensor] = None):
        backbone_features = self.forward_backbone(x)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses
        outputs, losses = self.forward_detect(backbone_features=backbone_features, targets=targets)
        return outputs, losses


class YoloXPEPRDetector(th.nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        # 1. メインのRGBストリーム (生徒)
        self.backbone = build_yolo_backbone(model_cfg.backbone)
        in_channels = self.backbone.get_stage_dims(model_cfg.fpn.in_stages)
        self.fpn = build_yolox_fpn(model_cfg.fpn, in_channels=in_channels)
        strides = self.backbone.get_strides(model_cfg.fpn.in_stages)
        self.yolox_head = build_yolox_head(model_cfg.head, in_channels=in_channels, strides=strides)

        model_cfg.event_predictor.in_channels = in_channels[-1]
        self.event_backbone = build_yolo_backbone(model_cfg.event_backbone)
        self.event_predictor = build_pepr_predictor(model_cfg.event_predictor) 

    def forward_backbone(self, x: th.Tensor):
        with CudaTimer(device=x.device, timer_name="Backbone"):
            return self.backbone(x)

    def forward_detect(self, backbone_features, targets: Optional[th.Tensor] = None):
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
            
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.yolox_head(fpn_features, targets)
            return outputs, losses
            
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.yolox_head(fpn_features)
        return outputs, losses

    def forward(self, 
                x: th.Tensor, 
                event_data: Optional[th.Tensor] = None,
                targets: Optional[th.Tensor] = None,
                retrieve_detections: bool = True):
        
        backbone_features = self.forward_backbone(x)
        outputs, losses = None, {}

        if self.training:
            assert targets is not None
            assert event_data is not None, "Training YoloXPEPRDetector requires 'event_data'."
            
            # YOLOXの通常のタスクロス
            outputs, task_losses = self.forward_detect(backbone_features, targets)
            losses.update(task_losses)

            # PEPR: イベント特徴の予測ロス
            target_stage = 5 

            with th.no_grad():
                event_features_dict = self.event_backbone(event_data)
                event_feat_map = event_features_dict[target_stage] # (B, C, H, W)
            
            # 1. アクティビティに基づいて M=4 個のパッチインデックスをサンプリング
            patch_indices = self.sample_patch_indices(event_feat_map, num_patches=4)
            
            # 2. RGB特徴量マップとターゲット位置を予測器に渡す
            rgb_feat_map = backbone_features[target_stage]
            event_pred = self.event_predictor(rgb_feat_map, patch_indices)
            
            # 3. 予測パッチと実際のイベントパッチ間のMSEロスを計算
            losses["loss_pepr_event"] = self.compute_pepr_loss(
                predicted_patches=event_pred, 
                target_features=event_feat_map, 
                patch_indices=patch_indices
            )

            return outputs, losses

        if not retrieve_detections:
            return None, None
            
        outputs, _ = self.forward_detect(backbone_features)
        return outputs, None

    def sample_patch_indices(self, event_feat_map: th.Tensor, num_patches: int = 4) -> th.Tensor:
        """
        イベント特徴マップのアクティビティに基づいてパッチ(位置)を選択します。
        戻り値: (B, num_patches) の平坦化された空間インデックス
        """
        B, C, H, W = event_feat_map.shape
        
        # チャンネル方向のL2ノルムを計算し、空間的な「アクティビティ」とする: (B, H, W)
        activity = event_feat_map.norm(dim=1)
        
        # 空間次元を平坦化: (B, H*W)
        activity_flat = activity.view(B, -1)
        
        # 各バッチに対して、アクティビティ上位と下位を半分ずつサンプリング
        half_k = num_patches // 2
        indices = []
        for b in range(B):
            # イベントが活発な領域 (動くエッジなど)
            topk_idx = th.topk(activity_flat[b], k=half_k).indices
            # イベントが静かな領域 (背景など)
            bottomk_idx = th.topk(activity_flat[b], k=num_patches - half_k, largest=False).indices
            
            # 結合して保存
            indices.append(th.cat([topk_idx, bottomk_idx]))
            
        return th.stack(indices) # Shape: (B, num_patches)

    def compute_pepr_loss(self, predicted_patches, target_features, patch_indices):
        """
        predicted_patches: (B, M, C) 予測器からの出力
        target_features: (B, C, H, W) イベントエンコーダからの正解特徴量マップ
        patch_indices: (B, M) サンプリングされた平坦化インデックス
        """
        B, C, H, W = target_features.shape
        
        # 1. 正解特徴量を平坦化: (B, C, H, W) -> (B, H*W, C)
        flat_targets = target_features.flatten(2).transpose(1, 2)
        
        # 2. サンプリングされたインデックスを使って、正解パッチ特徴を抽出
        # patch_indices を (B, M, C) の形に拡張して gather で一気に取り出す
        expanded_indices = patch_indices.unsqueeze(-1).expand(-1, -1, C)
        actual_patches = th.gather(flat_targets, dim=1, index=expanded_indices) # (B, M, C)
        
        # 3. 論文数式(2)の MSE Loss (平均二乗誤差) を計算
        loss = F.mse_loss(predicted_patches, actual_patches)
        
        return loss