import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class TPhaseNet(nn.Module):
    """TPhaseNet: Transformer-enhanced PhaseNet for seismic phase picking.

    PhaseNet의 U-Net 구조에 Transformer 블록을 결합하여
    P파, S파 도착 시각을 검출하는 딥러닝 모델.

    입력:  (B, 3, T) - 3성분 지진파형 (Z, N, E)
    출력:  (B, 3, T) - 각 시점별 [Noise, P파, S파] 확률
    """

    def __init__(self, in_channels=3, classes=3, filters_root=8, depth=8,
                 kernel_size=7, stride=4, transformer_start_level=4,
                 n_heads=4, ff_dim_factor=4, dropout=0.1,
                 activation="silu", skip_attention=True, lstm_hidden=64):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            filters_root=filters_root,
            depth=depth,
            kernel_size=kernel_size,
            stride=stride,
            transformer_start_level=transformer_start_level,
            n_heads=n_heads,
            ff_dim_factor=ff_dim_factor,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = Decoder(
            classes=classes,
            filters_root=filters_root,
            depth=depth,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            skip_attention=skip_attention,
            lstm_hidden=lstm_hidden,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, T) - 3성분 지진파형
        Returns:
            (B, 3, T) - [Noise, P, S] 확률 (softmax)
        """
        bottleneck, skips = self.encoder(x)
        output = self.decoder(bottleneck, skips)
        return output

    @classmethod
    def from_config(cls, config):
        """YAML 설정에서 모델 생성."""
        model_cfg = config["model"]
        return cls(
            in_channels=model_cfg.get("in_channels", 3),
            classes=model_cfg.get("classes", 3),
            filters_root=model_cfg.get("filters_root", 8),
            depth=model_cfg.get("depth", 8),
            kernel_size=model_cfg.get("kernel_size", 7),
            stride=model_cfg.get("stride", 4),
            transformer_start_level=model_cfg.get("transformer_start_level", 4),
            n_heads=model_cfg.get("n_heads", 4),
            ff_dim_factor=model_cfg.get("ff_dim_factor", 4),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "silu"),
            skip_attention=model_cfg.get("skip_attention", True),
            lstm_hidden=model_cfg.get("lstm_hidden", 64),
        )

    def count_parameters(self):
        """학습 가능한 파라미터 수 반환."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
