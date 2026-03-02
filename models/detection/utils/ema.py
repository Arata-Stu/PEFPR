import torch
import math
from copy import deepcopy

class ModelEMA:
    """
    シンプルな Model Exponential Moving Average
    """
    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): EMAを適用するモデル
            decay (float): EMAの基本減衰率
            updates (int): EMAの更新回数カウンタ
        """
        # EMAモデルを作成し、勾配計算を無効化
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.updates = updates
        # 初期エポックを助けるための動的減衰関数の定義
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.0))

    @torch.no_grad()
    def update(self, model):
        """メインモデルの重みを使ってEMAモデルを更新する"""
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1.0 - d) * msd[k].detach()

    def state_dict(self):
        """チェックポイント保存用"""
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        """チェックポイント復元用"""
        self.ema.load_state_dict(state_dict)