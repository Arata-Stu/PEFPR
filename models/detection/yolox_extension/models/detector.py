from typing import Dict, Optional, Tuple, Union

import torch as th
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
        
        # 1. RGBの順伝播
        backbone_features = self.forward_backbone(x)
        outputs, losses = None, {}

        # 2. 学習時のみ：タスクロスとイベント予測ロスの計算
        if self.training:
            assert targets is not None
            assert event_data is not None, "Training YoloXPEPRDetector requires 'event_data'."
            
            # YOLOXの通常のタスクロス
            outputs, task_losses = self.forward_detect(backbone_features, targets)
            losses.update(task_losses)

            # PEPR: イベント特徴の予測ロス
            with th.no_grad():
                event_features = self.event_backbone(event_data)
            
            event_pred = self.event_predictor(backbone_features)
            losses["loss_pepr_event"] = self.compute_pepr_loss(event_pred, event_features)

            return outputs, losses

        # 3. 推論時
        if not retrieve_detections:
            return None, None
            
        outputs, _ = self.forward_detect(backbone_features)
        return outputs, None

    def compute_pepr_loss(self, predicted_patches, target_features):
        """
        ここにパッチの抽出と平均二乗誤差(MSE)の計算を実装します。
        """
        # TODO: 実装
        pass