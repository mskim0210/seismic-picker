import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """사인/코사인 위치 인코딩.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=6000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, C) - batch_first 형식
        Returns:
            (B, T, C) - 위치 인코딩이 더해진 텐서
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Transformer Encoder Block for 1D feature maps.

    Conv 특징맵 (B, C, T) 형식을 받아 self-attention을 적용하고
    동일한 형식으로 출력. Residual connection 포함.
    """

    def __init__(self, d_model, n_heads=4, ff_dim_factor=4, dropout=0.1):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_dim_factor,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) - conv feature map
        Returns:
            (B, C, T) - transformer 적용 + residual
        """
        # (B, C, T) -> (B, T, C)
        residual = x
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer_layer(x)
        x = self.norm(x)
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        return x + residual
