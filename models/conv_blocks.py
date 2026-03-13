import torch
import torch.nn as nn


def _get_activation(activation="silu"):
    """activation 문자열로 nn.Module 반환."""
    if activation == "silu":
        return nn.SiLU(inplace=True)
    return nn.ReLU(inplace=True)


class DownBlock(nn.Module):
    """인코더 다운샘플링 블록 (ResNet).

    Conv1d(same) -> BN -> Act -> Conv1d(stride) -> BN + residual -> Act
    Projection shortcut: 1x1 Conv(stride) -> BN
    skip connection은 다운샘플링 전 출력을 반환.
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4,
                 activation="silu"):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = _get_activation(activation)

        # Projection shortcut: in_channels → out_channels, T → T//stride
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T)
        Returns:
            skip: (B, C_out, T) - 다운샘플링 전 특징 (skip connection용)
            out:  (B, C_out, T//stride) - residual 적용된 다운샘플링 결과
        """
        identity = x

        x = self.act(self.bn1(self.conv1(x)))
        skip = x
        out = self.bn2(self.conv2(x))

        out = self.act(out + self.shortcut(identity))
        return skip, out


class UpBlock(nn.Module):
    """디코더 업샘플링 블록 (ResNet).

    ConvTranspose1d(stride) -> BN -> Act -> concat(skip) -> Conv1d -> BN + residual -> Act
    Projection shortcut: 1x1 Conv -> BN (concat 채널 → out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4,
                 activation="silu"):
        super().__init__()
        padding = kernel_size // 2

        self.upconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=stride - 1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # concat 후 채널 수: out_channels(업샘플) + out_channels(skip)
        self.conv = nn.Conv1d(out_channels * 2, out_channels, kernel_size,
                              padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = _get_activation(activation)

        # Projection shortcut: concat 채널(out*2) → out_channels
        self.shortcut = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x, skip):
        """
        Args:
            x:    (B, C_in, T_small)
            skip: (B, C_out, T_large) - 인코더에서 온 skip connection
        Returns:
            out:  (B, C_out, T_large)
        """
        x = self.act(self.bn1(self.upconv(x)))

        # 업샘플링 결과와 skip 길이가 다를 수 있으므로 맞춤
        diff = skip.size(2) - x.size(2)
        if diff > 0:
            x = nn.functional.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :skip.size(2)]

        concat = torch.cat([x, skip], dim=1)
        out = self.bn2(self.conv(concat))

        out = self.act(out + self.shortcut(concat))
        return out
