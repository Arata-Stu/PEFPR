import numpy as np
import torch

class EventRepresentationBase:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, events: dict) -> torch.Tensor:
        raise NotImplementedError

class EventHistogram(EventRepresentationBase):
    """正極・負極の2チャネルヒストグラム (2, H, W)"""
    def __call__(self, events: dict) -> torch.Tensor:
        img = np.zeros((2, self.height, self.width), dtype=np.float32)
        
        if len(events['x']) > 0:
            x = events['x'].astype(int)
            y = events['y'].astype(int)
            p = events['p'].flatten()
            
            # 負極 (p <= 0) と 正極 (p > 0) のマスク
            mask_neg = (p <= 0)
            mask_pos = (p > 0)
            
            # 各ピクセルのイベント発生回数をカウント
            np.add.at(img[0], (y[mask_neg], x[mask_neg]), 1)
            np.add.at(img[1], (y[mask_pos], x[mask_pos]), 1)
            
        return torch.from_numpy(img)

class VoxelGrid(EventRepresentationBase):
    """時間をB個のビンに分割した時空間ボクセルグリッド (B, H, W)"""
    def __init__(self, height: int, width: int, num_bins: int = 5):
        super().__init__(height, width)
        self.num_bins = num_bins

    def __call__(self, events: dict) -> torch.Tensor:
        voxel_grid = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        
        if len(events['x']) > 0:
            x = events['x'].astype(int)
            y = events['y'].astype(int)
            # 極性を -1 と 1 に変換
            p = np.where(events['p'].flatten() > 0, 1.0, -1.0)
            t = events['t']
            
            # 時間を [0, num_bins - 1] の範囲に正規化
            t_min, t_max = t.min(), t.max()
            if t_max > t_min:
                t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1)
            else:
                t_norm = np.zeros_like(t)
                
            t_idx = np.clip(t_norm.astype(int), 0, self.num_bins - 1)
            
            # ボクセルに極性を加算
            np.add.at(voxel_grid, (t_idx, y, x), p)
            
        return torch.from_numpy(voxel_grid)