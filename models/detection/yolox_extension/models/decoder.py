import torch
import torch.nn as nn

class PEPRTransformerDecoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 num_patches: int = 4, 
                 num_layers: int = 4,   
                 num_heads: int = 8): 
        super().__init__()
        
        self.num_patches = num_patches
        self.in_channels = in_channels
        
        # 1. Transformerデコーダ層の定義 
        # batch_first=True にすることで (Batch, Sequence, Features) の形を扱えるようにします
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # 2. クエリ用の学習可能な位置埋め込み（Positional Embeddings） 
        # M個のパッチそれぞれに対応するクエリを用意します
        self.query_embed = nn.Embedding(num_patches, in_channels)

    def forward(self, rgb_features: torch.Tensor):
        """
        引数:
            rgb_features: YOLOXバックボーンから出力された特徴量マップ dict[key, value] 形式で、valueのshapeは (B, C, H, W)
        戻り値:
            predicted_patches: 予測されたM個のイベント特徴。Shape: (B, M, C)
        """
        # ステージ5の特徴量を利用
        rgb_feature = rgb_features[5]  # 例: (B, C, H, W)
        B, C, H, W = rgb_feature.shape
        
        # 1. RGB特徴量を平坦化 (Flatten) して系列データにする 
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        memory = rgb_feature.flatten(2).transpose(1, 2)
        
        # 2. Transformerに入力するクエリの準備 
        # Embeddingの重みを全バッチで共有して展開します
        # Shape: (B, num_patches, C)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # 3. Transformerデコーダによる予測計算 
        # tgt (ターゲット/クエリ): 位置埋め込み
        # memory (コンテキスト): 平坦化されたRGB特徴量
        predicted_patches = self.transformer_decoder(tgt=queries, memory=memory)
        
        return predicted_patches