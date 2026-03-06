import torch
import torch.nn as nn

from .conv_blocks import UpBlock


class Decoder(nn.Module):
    """TPhaseNet 디코더.

    인코더의 역순으로 업샘플링하며, skip connection을 결합.
    마지막에 1x1 conv + softmax로 클래스별 확률 출력.

    입력:  bottleneck (B, C_bottleneck, T_small), skips list
    출력:  (B, classes, T_original)
    """

    def __init__(self, classes=3, filters_root=8, depth=8,
                 kernel_size=7, stride=4):
        super().__init__()
        self.depth = depth

        # 업샘플링 블록 (역순: depth-2 ~ 0)
        self.up_blocks = nn.ModuleList()
        for level in range(depth - 2, -1, -1):
            if level == depth - 2:
                ch_in = filters_root * (2 ** (depth - 1))
            else:
                ch_in = filters_root * (2 ** (level + 1))

            ch_out = filters_root * (2 ** level) if level > 0 else filters_root
            self.up_blocks.append(
                UpBlock(ch_in, ch_out, kernel_size, stride)
            )

        # 출력 헤드: 1x1 conv로 클래스 수만큼 채널 생성
        self.output_conv = nn.Conv1d(filters_root, classes, kernel_size=1)

    def forward(self, x, skips):
        """
        Args:
            x: (B, C_bottleneck, T_small) - 인코더 bottleneck 출력
            skips: list of (B, C_level, T_level) - 인코더 skip connections
                   순서: [level0, level1, ..., level(depth-2)]
        Returns:
            (B, classes, T_original) - softmax 확률
        """
        # skips를 역순으로 사용 (깊은 레벨부터)
        for i, up_block in enumerate(self.up_blocks):
            skip_idx = len(skips) - 1 - i
            x = up_block(x, skips[skip_idx])

        x = self.output_conv(x)
        x = torch.softmax(x, dim=1)
        return x
