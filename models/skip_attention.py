import torch
import torch.nn as nn


class SkipAttentionBlock(nn.Module):
    """Skip connection에 BiLSTM + Cross-Attention을 적용하는 블록.

    NORSAR TPhasenet의 "across" attention 방식을 참고.
    Encoder skip feature에 BiLSTM으로 temporal context를 학습한 후,
    decoder feature와 cross-attention을 수행한다.

    입력:
        skip:        (B, C, T) - encoder skip feature
        decoder_feat: (B, C, T) - upsampled decoder feature (skip과 동일 해상도)
    출력:
        attended_skip: (B, C, T) - attention 적용된 skip feature
    """

    def __init__(self, channels, lstm_hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.channels = channels

        # BiLSTM: temporal context 학습
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 1x1 Conv projection: BiLSTM output(lstm_hidden*2) → channels
        self.proj = nn.Conv1d(lstm_hidden * 2, channels, kernel_size=1)
        self.proj_norm = nn.LayerNorm(channels)

        # Cross-attention: query=projected skip, key/value=decoder feature
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(channels)

    def forward(self, skip, decoder_feat):
        """
        Args:
            skip:         (B, C, T) - encoder skip connection feature
            decoder_feat: (B, C, T) - decoder upsampled feature (same T as skip)
        Returns:
            (B, C, T) - attended skip feature + residual
        """
        residual = skip

        # BiLSTM: (B, C, T) → (B, T, C) → LSTM → (B, T, lstm_hidden*2)
        s = skip.permute(0, 2, 1)
        s, _ = self.lstm(s)

        # 1x1 Conv projection: (B, T, H*2) → (B, H*2, T) → (B, C, T) → (B, T, C)
        s = s.permute(0, 2, 1)
        s = self.proj(s)
        query = self.proj_norm(s.permute(0, 2, 1))

        # Cross-attention
        kv = decoder_feat.permute(0, 2, 1)
        attended, _ = self.cross_attn(query, kv, kv)
        attended = self.attn_norm(attended)

        # (B, T, C) → (B, C, T) + residual
        return attended.permute(0, 2, 1) + residual
