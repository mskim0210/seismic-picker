import torch
import torch.nn as nn

from .conv_blocks import UpBlock
from .skip_attention import SkipAttentionBlock


class Decoder(nn.Module):
    """TPhaseNet 디코더.

    인코더의 역순으로 업샘플링하며, skip connection을 결합.
    skip_attention=True일 때 BiLSTM + Cross-Attention으로 skip을 처리.
    마지막에 1x1 conv + softmax로 클래스별 확률 출력.

    입력:  bottleneck (B, C_bottleneck, T_small), skips list
    출력:  (B, classes, T_original)
    """

    def __init__(self, classes=3, filters_root=8, depth=8,
                 kernel_size=7, stride=4, activation="silu",
                 skip_attention=True, lstm_hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.use_skip_attention = skip_attention

        # 업샘플링 블록 (역순: depth-2 ~ 0)
        self.up_blocks = nn.ModuleList()
        self.skip_attn_blocks = nn.ModuleList() if skip_attention else None

        for level in range(depth - 2, -1, -1):
            if level == depth - 2:
                ch_in = filters_root * (2 ** (depth - 1))
            else:
                ch_in = filters_root * (2 ** (level + 1))

            ch_out = filters_root * (2 ** level) if level > 0 else filters_root
            self.up_blocks.append(
                UpBlock(ch_in, ch_out, kernel_size, stride,
                        activation=activation)
            )

            if skip_attention:
                self.skip_attn_blocks.append(
                    SkipAttentionBlock(ch_out, lstm_hidden, n_heads, dropout)
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
        for i, up_block in enumerate(self.up_blocks):
            skip_idx = len(skips) - 1 - i
            skip = skips[skip_idx]

            if self.use_skip_attention:
                # upconv로 decoder feature를 skip과 동일 해상도로 업샘플링
                decoder_feat = up_block.act(up_block.bn1(up_block.upconv(x)))

                # 길이 맞춤
                diff = skip.size(2) - decoder_feat.size(2)
                if diff > 0:
                    decoder_feat = nn.functional.pad(decoder_feat, (0, diff))
                elif diff < 0:
                    decoder_feat = decoder_feat[:, :, :skip.size(2)]

                # Skip attention 적용
                attended_skip = self.skip_attn_blocks[i](skip, decoder_feat)

                # concat + conv + residual (UpBlock 후반부)
                concat = torch.cat([decoder_feat, attended_skip], dim=1)
                out = up_block.bn2(up_block.conv(concat))
                x = up_block.act(out + up_block.shortcut(concat))
            else:
                x = up_block(x, skip)

        x = self.output_conv(x)
        x = torch.softmax(x, dim=1)
        return x
