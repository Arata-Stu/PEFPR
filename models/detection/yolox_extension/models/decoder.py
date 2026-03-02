import torch
import torch.nn as nn

class PEPRTransformerPredictor(nn.Module):
    def __init__(self, in_channels: int, num_layers: int = 4, num_heads: int = 8, max_spatial_size: int = 1600):
        super().__init__()
        # (B, H*W, C) を処理できるTransformerデコーダ
        decoder_layer = nn.TransformerDecoderLayer(d_model=in_channels, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 空間全体の学習可能な位置埋め込み (Positional Embeddings)
        # max_spatial_size は H*W の最大値 (例: dark5 が 40x40 なら 1600)
        self.pos_embed = nn.Parameter(torch.randn(1, max_spatial_size, in_channels))

    def forward(self, rgb_features: torch.Tensor, patch_indices: torch.Tensor):
        """
        rgb_features: (B, C, H, W) RGBのバックボーン特徴量 (dark5)
        patch_indices: (B, M) 予測すべき空間位置のインデックス
        """
        B, C, H, W = rgb_features.shape
        M = patch_indices.shape[1]
        
        # 1. RGB特徴量を系列に変換し、位置情報を付与 (Memory)
        memory = rgb_features.flatten(2).transpose(1, 2) # (B, H*W, C)
        memory = memory + self.pos_embed[:, :H*W, :]     # 位置埋め込みを加算
        
        # 2. 予測したいターゲット位置の位置埋め込みを Query として抽出
        # patch_indices (B, M) を (B, M, C) に拡張
        expanded_indices = patch_indices.unsqueeze(-1).expand(-1, -1, C)
        pos_embed_batch = self.pos_embed[:, :H*W, :].expand(B, -1, -1)
        
        # 指定された場所の Positional Embedding だけを抽出 -> これが「ここを予測して」という質問(Query)になる
        queries = torch.gather(pos_embed_batch, dim=1, index=expanded_indices) # (B, M, C)
        
        # 3. デコーダで予測
        predicted_patches = self.transformer_decoder(tgt=queries, memory=memory) # (B, M, C)
        
        return predicted_patches