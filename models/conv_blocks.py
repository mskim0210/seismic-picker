import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """인코더 다운샘플링 블록.

    Conv1d(same) -> BN -> ReLU -> Conv1d(stride) -> BN -> ReLU
    skip connection은 다운샘플링 전 출력을 반환.
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T)
        Returns:
            skip: (B, C_out, T) - 다운샘플링 전 특징 (skip connection용)
            out:  (B, C_out, T//stride) - 다운샘플링된 특징
        """
        x = self.relu(self.bn1(self.conv1(x)))
        skip = x
        out = self.relu(self.bn2(self.conv2(x)))
        return skip, out


class UpBlock(nn.Module):
    """디코더 업샘플링 블록.

    ConvTranspose1d(stride) -> BN -> ReLU -> concat(skip) -> Conv1d -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4):
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

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        """
        Args:
            x:    (B, C_in, T_small)
            skip: (B, C_out, T_large) - 인코더에서 온 skip connection
        Returns:
            out:  (B, C_out, T_large)
        """
        x = self.relu(self.bn1(self.upconv(x)))

        # 업샘플링 결과와 skip 길이가 다를 수 있으므로 맞춤
        diff = skip.size(2) - x.size(2)
        if diff > 0:
            x = nn.functional.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :skip.size(2)]

        x = torch.cat([x, skip], dim=1)
        out = self.relu(self.bn2(self.conv(x)))
        return out
