import torch.nn as nn

from .conv_blocks import DownBlock
from .transformer_block import TransformerBlock


class Encoder(nn.Module):
    """TPhaseNet 인코더.

    7개의 다운샘플링 레벨 + 1개의 bottleneck으로 구성.
    Level transformer_start_level 이상에서 Transformer 블록 적용.

    입력:  (B, in_channels, T)
    출력:  bottleneck (B, C_bottleneck, T_small), skips list
    """

    def __init__(self, in_channels=3, filters_root=8, depth=8,
                 kernel_size=7, stride=4, transformer_start_level=4,
                 n_heads=4, ff_dim_factor=4, dropout=0.1,
                 activation="silu"):
        super().__init__()
        self.depth = depth
        self.transformer_start_level = transformer_start_level

        from .conv_blocks import _get_activation

        # 입력 컨볼루션: in_channels -> filters_root
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, filters_root, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(filters_root),
            _get_activation(activation),
        )

        # 다운샘플링 블록 (levels 0 ~ depth-2)
        self.down_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleDict()

        for level in range(depth - 1):
            # 채널 수 계산: level 0은 root, level i는 root * 2^i
            if level == 0:
                ch_in = filters_root
                ch_out = filters_root
            else:
                ch_in = filters_root * (2 ** (level - 1))
                ch_out = filters_root * (2 ** level)

            self.down_blocks.append(
                DownBlock(ch_in, ch_out, kernel_size, stride,
                          activation=activation)
            )

            # Transformer 블록 (해당 레벨에만)
            if level >= transformer_start_level:
                heads = n_heads if level < depth - 2 else n_heads * 2
                self.transformer_blocks[str(level)] = TransformerBlock(
                    ch_out, heads, ff_dim_factor, dropout
                )

        # Bottleneck (마지막 레벨): conv + transformer, 다운샘플링 없음
        bottleneck_in = filters_root * (2 ** (depth - 2))
        bottleneck_out = filters_root * (2 ** (depth - 1))
        self.bottleneck_conv = nn.Sequential(
            nn.Conv1d(bottleneck_in, bottleneck_out, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(bottleneck_out),
            _get_activation(activation),
        )
        self.bottleneck_transformer = TransformerBlock(
            bottleneck_out, n_heads * 2, ff_dim_factor, dropout
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, T)
        Returns:
            bottleneck: (B, C_bottleneck, T_small)
            skips: list of tensors, 인코더 각 레벨의 skip connection
        """
        x = self.input_conv(x)
        skips = []

        for level in range(self.depth - 1):
            skip, x = self.down_blocks[level](x)

            # Transformer 블록 적용 (skip connection 이후 다운샘플링 전 특징에 적용)
            level_key = str(level)
            if level_key in self.transformer_blocks:
                skip = self.transformer_blocks[level_key](skip)

            skips.append(skip)

        # Bottleneck
        x = self.bottleneck_conv(x)
        x = self.bottleneck_transformer(x)

        return x, skips
