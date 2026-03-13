# Design: NORSAR 기반 모델 개선

> Plan 문서: `docs/01-plan/features/model-improvements.plan.md`

## 1. 개요

NORSAR TPhasenet 논문의 핵심 기법 4가지를 seismic-picker에 incremental 방식으로 통합한다.

| Step | 개선사항 | 변경 파일 | 신규 파일 |
|------|---------|----------|----------|
| 1 | Swish(SiLU) activation | `conv_blocks.py`, `encoder.py`, `tphasenet.py`, `decoder.py`, config | - |
| 2 | ResNet block | `conv_blocks.py` | - |
| 3 | Skip connection attention | `decoder.py`, `tphasenet.py`, config | `models/skip_attention.py` |
| 4 | Gap insertion augmentation | `augmentation.py` | - |

---

## 2. Step 1: Swish(SiLU) Activation

### 2.1 변경 사항

`nn.ReLU` → `nn.SiLU` 교체. PyTorch 내장 `nn.SiLU`는 Swish activation (`x * sigmoid(x)`)과 동일하다.

### 2.2 `conv_blocks.py` 변경

```python
class DownBlock(nn.Module):
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

        self.act = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        skip = x
        out = self.act(self.bn2(self.conv2(x)))
        return skip, out


class UpBlock(nn.Module):
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

        self.conv = nn.Conv1d(out_channels * 2, out_channels, kernel_size,
                              padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.act(self.bn1(self.upconv(x)))
        # ... (길이 맞춤 로직 동일)
        x = torch.cat([x, skip], dim=1)
        out = self.act(self.bn2(self.conv(x)))
        return out
```

### 2.3 `encoder.py` 변경

```python
class Encoder(nn.Module):
    def __init__(self, ..., activation="silu"):
        # activation 파라미터를 저장하고 DownBlock에 전달
        act_fn = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, filters_root, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters_root),
            act_fn,
        )

        # DownBlock 생성 시 activation 전달
        self.down_blocks.append(
            DownBlock(ch_in, ch_out, kernel_size, stride, activation=activation)
        )

        # bottleneck_conv에도 동일 적용
        self.bottleneck_conv = nn.Sequential(
            nn.Conv1d(bottleneck_in, bottleneck_out, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(bottleneck_out),
            nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True),
        )
```

### 2.4 `decoder.py` 변경

```python
class Decoder(nn.Module):
    def __init__(self, ..., activation="silu"):
        # UpBlock 생성 시 activation 전달
        self.up_blocks.append(
            UpBlock(ch_in, ch_out, kernel_size, stride, activation=activation)
        )
```

### 2.5 `tphasenet.py` 변경

```python
class TPhaseNet(nn.Module):
    def __init__(self, ..., activation="silu"):
        self.encoder = Encoder(..., activation=activation)
        self.decoder = Decoder(..., activation=activation)

    @classmethod
    def from_config(cls, config):
        model_cfg = config["model"]
        return cls(
            ...,
            activation=model_cfg.get("activation", "silu"),
        )
```

### 2.6 Config 변경

`config/default.yaml`:
```yaml
model:
  activation: "silu"
```

`config/defaults.py`:
```python
"activation": "silu",
```

---

## 3. Step 2: ResNet Block

### 3.1 설계 원리

NORSAR TPhasenet의 pre-activation residual block 방식을 참고하되, 현재 모델의 post-activation 패턴(Conv→BN→Act)을 유지하면서 residual path를 추가한다.

핵심: `output = F(x) + shortcut(x)` 형태로, 입력과 출력의 채널/시간이 다를 때 projection shortcut(1x1 Conv1d)을 사용한다.

### 3.2 `DownBlock` 변경

```python
class DownBlock(nn.Module):
    """인코더 다운샘플링 블록 (ResNet).

    Conv1d(same) -> BN -> Act -> Conv1d(stride) -> BN -> Act
    + residual: 1x1 Conv(stride) -> BN (projection shortcut)
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

        self.act = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

        # Projection shortcut: in_channels → out_channels, T → T//stride
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        """
        Returns:
            skip: (B, C_out, T) - 다운샘플링 전 특징
            out:  (B, C_out, T//stride) - residual 적용된 다운샘플링 결과
        """
        identity = x

        x = self.act(self.bn1(self.conv1(x)))
        skip = x
        out = self.bn2(self.conv2(x))

        # Residual: shortcut으로 identity 차원 맞춤
        out = self.act(out + self.shortcut(identity))
        return skip, out
```

**Tensor shape 흐름 (depth=5, stride=2, filters_root=8):**

| Level | Input | conv1 out (=skip) | conv2 out | shortcut | residual out |
|-------|-------|-------------------|-----------|----------|-------------|
| 0 | (B,8,6000) | (B,8,6000) | (B,8,3000) | (B,8,3000) | (B,8,3000) |
| 1 | (B,8,3000) | (B,16,3000) | (B,16,1500) | (B,16,1500) | (B,16,1500) |
| 2 | (B,16,1500) | (B,32,1500) | (B,32,750) | (B,32,750) | (B,32,750) |
| 3 | (B,32,750) | (B,64,750) | (B,64,375) | (B,64,375) | (B,64,375) |

### 3.3 `UpBlock` 변경

```python
class UpBlock(nn.Module):
    """디코더 업샘플링 블록 (ResNet).

    ConvTranspose1d(stride) -> BN -> Act -> concat(skip) -> Conv1d -> BN -> Act
    + residual: concat 후 1x1 Conv -> BN (projection shortcut)
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

        self.conv = nn.Conv1d(out_channels * 2, out_channels, kernel_size,
                              padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

        # Projection shortcut: concat 후 채널(out*2) → out_channels
        self.shortcut = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x, skip):
        x = self.act(self.bn1(self.upconv(x)))

        # 길이 맞춤
        diff = skip.size(2) - x.size(2)
        if diff > 0:
            x = nn.functional.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :skip.size(2)]

        concat = torch.cat([x, skip], dim=1)
        out = self.bn2(self.conv(concat))

        # Residual
        out = self.act(out + self.shortcut(concat))
        return out
```

---

## 4. Step 3: Skip Connection Attention

### 4.1 설계 원리

NORSAR TPhasenet의 `"across"` 모드를 참고한다. Encoder skip feature에 BiLSTM으로 temporal context를 학습한 후, MultiheadAttention으로 decoder feature와 cross-attention을 수행한다.

### 4.2 `models/skip_attention.py` (신규)

```python
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
        # input: (B, T, C), output: (B, T, lstm_hidden*2)
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 1x1 Conv projection: BiLSTM output → channels 차원으로 맞춤
        self.proj = nn.Conv1d(lstm_hidden * 2, channels, kernel_size=1)
        self.proj_norm = nn.LayerNorm(channels)

        # Cross-attention
        # query: projected skip (encoder), key/value: decoder feature
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
        s = skip.permute(0, 2, 1)  # (B, T, C)
        s, _ = self.lstm(s)         # (B, T, lstm_hidden*2)

        # 1x1 Conv projection: (B, T, lstm_hidden*2) → (B, lstm_hidden*2, T) → (B, C, T)
        s = s.permute(0, 2, 1)      # (B, lstm_hidden*2, T)
        s = self.proj(s)             # (B, C, T)

        # Cross-attention: (B, C, T) → (B, T, C) for attention
        query = self.proj_norm(s.permute(0, 2, 1))       # (B, T, C)
        kv = decoder_feat.permute(0, 2, 1)                # (B, T, C)

        attended, _ = self.cross_attn(query, kv, kv)       # (B, T, C)
        attended = self.attn_norm(attended)

        # (B, T, C) → (B, C, T) + residual
        out = attended.permute(0, 2, 1) + residual
        return out
```

### 4.3 `decoder.py` 변경

```python
import torch
import torch.nn as nn

from .conv_blocks import UpBlock
from .skip_attention import SkipAttentionBlock


class Decoder(nn.Module):
    def __init__(self, classes=3, filters_root=8, depth=8,
                 kernel_size=7, stride=4, activation="silu",
                 skip_attention=True, lstm_hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.use_skip_attention = skip_attention

        self.up_blocks = nn.ModuleList()
        self.skip_attn_blocks = nn.ModuleList() if skip_attention else None

        for level in range(depth - 2, -1, -1):
            if level == depth - 2:
                ch_in = filters_root * (2 ** (depth - 1))
            else:
                ch_in = filters_root * (2 ** (level + 1))

            ch_out = filters_root * (2 ** level) if level > 0 else filters_root
            self.up_blocks.append(
                UpBlock(ch_in, ch_out, kernel_size, stride, activation=activation)
            )

            if skip_attention:
                self.skip_attn_blocks.append(
                    SkipAttentionBlock(ch_out, lstm_hidden, n_heads, dropout)
                )

        self.output_conv = nn.Conv1d(filters_root, classes, kernel_size=1)

    def forward(self, x, skips):
        for i, up_block in enumerate(self.up_blocks):
            skip_idx = len(skips) - 1 - i

            if self.use_skip_attention:
                # UpBlock 내부에서 upconv 후의 x가 필요하므로
                # 먼저 x를 upsample하여 decoder_feat를 생성
                # UpBlock.forward를 분할하는 대신,
                # skip에 attention을 적용할 때 decoder_feat로 x를 사용
                #
                # 단, x의 시간 해상도가 skip과 다를 수 있으므로
                # upconv만 먼저 수행하여 해상도를 맞춘 후 attention 적용
                decoder_feat = up_block.act(up_block.bn1(up_block.upconv(x)))

                # 길이 맞춤
                skip = skips[skip_idx]
                diff = skip.size(2) - decoder_feat.size(2)
                if diff > 0:
                    decoder_feat = nn.functional.pad(decoder_feat, (0, diff))
                elif diff < 0:
                    decoder_feat = decoder_feat[:, :, :skip.size(2)]

                # Skip attention 적용
                attended_skip = self.skip_attn_blocks[i](skip, decoder_feat)

                # concat + conv + residual (UpBlock의 후반부)
                concat = torch.cat([decoder_feat, attended_skip], dim=1)
                out = up_block.bn2(up_block.conv(concat))
                x = up_block.act(out + up_block.shortcut(concat))
            else:
                x = up_block(x, skips[skip_idx])

        x = self.output_conv(x)
        x = torch.softmax(x, dim=1)
        return x
```

**핵심 설계 결정**: `skip_attention=True`일 때 `UpBlock.forward()`를 직접 호출하지 않고, UpBlock의 내부 레이어를 단계별로 호출한다. 이는 decoder_feat(upsampled x)를 cross-attention의 key/value로 사용해야 하기 때문이다.

### 4.4 Tensor Shape 흐름 (depth=5, skip_attention=True)

```
Decoder level 3 (가장 깊은 skip):
  x:            (B, 128, 375)     ← bottleneck
  upconv(x):    (B, 64, 750)      ← decoder_feat
  skip[3]:      (B, 64, 750)      ← encoder skip
  SkipAttention:
    lstm(skip):   (B, 750, 128)   ← BiLSTM output (64*2)
    proj:         (B, 64, 750)    ← 1x1 conv projection
    cross_attn:   (B, 750, 64)    ← query=proj(skip), kv=decoder_feat
    output:       (B, 64, 750)    ← attended skip + residual
  concat:       (B, 128, 750)     ← [decoder_feat, attended_skip]
  conv+res:     (B, 64, 750)      ← UpBlock 후반부

Decoder level 2:
  x:            (B, 64, 750)
  upconv(x):    (B, 32, 1500)
  skip[2]:      (B, 32, 1500)
  ... (동일 패턴)

Decoder level 1:
  x:            (B, 32, 1500)
  upconv(x):    (B, 16, 3000)
  skip[1]:      (B, 16, 3000)
  ...

Decoder level 0:
  x:            (B, 16, 3000)
  upconv(x):    (B, 8, 6000)
  skip[0]:      (B, 8, 6000)
  ...
  output_conv:  (B, 3, 6000)      ← final output
```

### 4.5 `tphasenet.py` 변경

```python
class TPhaseNet(nn.Module):
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

    @classmethod
    def from_config(cls, config):
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
```

### 4.6 Config 변경

`config/default.yaml` 추가:
```yaml
model:
  skip_attention: true
  lstm_hidden: 64
```

`config/defaults.py` 추가:
```python
"skip_attention": True,
"lstm_hidden": 64,
```

---

## 5. Step 4: Gap Insertion Augmentation

### 5.1 `augmentation.py`에 추가

```python
class RandomGapInsertion:
    """랜덤 데이터 결측 구간 삽입.

    실제 계측기 데이터에서 발생하는 데이터 드롭아웃/결측을 시뮬레이션.
    NORSAR TPhasenet 논문의 gap insertion augmentation 참고.

    Args:
        gap_length_range: (min, max) 갭 길이 (samples)
        max_gaps: 한 trace 내 최대 갭 수
        probability: 적용 확률
    """

    def __init__(self, gap_length_range=(50, 500), max_gaps=3, probability=0.2):
        self.gap_length_range = gap_length_range
        self.max_gaps = max_gaps
        self.probability = probability

    def __call__(self, waveform, labels):
        if np.random.random() > self.probability:
            return waveform, labels

        n_samples = waveform.shape[-1]
        n_gaps = np.random.randint(1, self.max_gaps + 1)

        for _ in range(n_gaps):
            gap_len = np.random.randint(*self.gap_length_range)
            max_start = n_samples - gap_len
            if max_start <= 0:
                continue
            start = np.random.randint(0, max_start)
            end = start + gap_len

            # 갭 구간을 0으로 채움
            waveform[:, start:end] = 0.0

            # 갭 구간의 라벨을 Noise로 설정
            labels[:, start:end] = 0.0
            labels[0, start:end] = 1.0  # Noise channel

        return waveform, labels
```

### 5.2 `get_default_augmentation()` 변경

```python
def get_default_augmentation():
    """기본 데이터 증강 파이프라인 반환."""
    return Compose([
        AddGaussianNoise(snr_range=(5, 30), probability=0.5),
        AmplitudeScale(scale_range=(0.5, 2.0), probability=0.5),
        RandomTimeShift(max_shift=200, probability=0.3),
        RandomChannelDrop(probability=0.1),
        RandomPolarityFlip(probability=0.5),
        RandomGapInsertion(gap_length_range=(50, 500), max_gaps=3, probability=0.2),
    ])
```

---

## 6. Config 전체 변경 요약

### 6.1 `config/default.yaml` (최종)

```yaml
model:
  name: "TPhaseNet"
  in_channels: 3
  classes: 3
  filters_root: 8
  depth: 5
  kernel_size: 7
  stride: 2
  transformer_start_level: 3
  n_heads: 4
  ff_dim_factor: 4
  dropout: 0.1
  activation: "silu"          # NEW: "silu" (Swish) or "relu"
  skip_attention: true         # NEW: Skip connection attention
  lstm_hidden: 64              # NEW: BiLSTM hidden size
```

### 6.2 `config/defaults.py` (최종 model 섹션)

```python
"model": {
    "name": "TPhaseNet",
    "in_channels": 3,
    "classes": 3,
    "filters_root": 8,
    "depth": 5,
    "kernel_size": 7,
    "stride": 2,
    "transformer_start_level": 3,
    "n_heads": 4,
    "ff_dim_factor": 4,
    "dropout": 0.1,
    "activation": "silu",          # NEW
    "skip_attention": True,        # NEW
    "lstm_hidden": 64,             # NEW
},
```

---

## 7. 테스트 전략

### 7.1 기존 테스트 호환성

`tests/conftest.py`의 `model_config` fixture에 새 파라미터가 없어도 기본값이 적용되므로 기존 82개 테스트가 통과해야 한다. 단, `UpBlock`의 `forward()` 시그니처가 변경되지 않으므로 (skip_attention 분기는 Decoder 내부에서 처리) 기존 `TestUpBlock`, `TestDownBlock` 테스트는 수정 없이 통과해야 한다.

**주의**: `DownBlock`과 `UpBlock` 생성자에 `activation` 파라미터가 추가되지만 기본값 `"silu"`가 설정되므로, 기존 테스트에서 파라미터 없이 생성하는 코드도 동작한다.

### 7.2 추가 테스트 항목

각 Step 구현 후 검증할 항목:

**Step 1 (Swish)**:
- `DownBlock(activation="silu")` / `DownBlock(activation="relu")` 모두 동일 shape 출력
- `TPhaseNet.from_config()`에서 activation 전달 확인

**Step 2 (ResNet)**:
- `DownBlock` residual: `out.shape == shortcut(identity).shape` 확인
- `UpBlock` residual: `out.shape == shortcut(concat).shape` 확인
- gradient flow: `loss.backward()` 후 모든 파라미터에 grad 존재

**Step 3 (Skip Attention)**:
- `SkipAttentionBlock` 단독: 입출력 shape (B, C, T) 동일
- `Decoder(skip_attention=True)`: 전체 forward shape 동일
- `Decoder(skip_attention=False)`: 기존 동작과 동일

**Step 4 (Gap Insertion)**:
- gap 구간의 waveform이 0인지 확인
- gap 구간의 labels[0] (Noise)이 1.0인지 확인
- probability=0일 때 변경 없음

---

## 8. 파라미터 수 예측

현재 모델 (depth=5, stride=2, filters_root=8): **~517K params**

추가되는 파라미터:

| 컴포넌트 | 추가 파라미터 (추정) |
|----------|-------------------|
| Swish activation | 0 (파라미터 없음) |
| ResNet shortcut (DownBlock x4) | ~(8*8 + 16*8 + 32*16 + 64*32) + BN = ~3.5K |
| ResNet shortcut (UpBlock x4) | ~(16*8 + 32*16 + 64*32 + 128*64) + BN = ~11K |
| SkipAttention BiLSTM (x4 levels) | ~4 * (C*64*4 + 64*64*4) = ~130K (가변) |
| SkipAttention projection (x4) | ~4 * (128*C) = ~20K |
| SkipAttention cross-attn (x4) | ~4 * (3*C*C) = ~50K |

**예상 총 파라미터**: ~730K (기존 대비 ~41% 증가)

> BiLSTM이 가장 큰 파라미터 증가 요인. lstm_hidden=64가 부담되면 32로 축소 가능.

---

## 9. 구현 순서 체크리스트

- [ ] **Step 1: Swish Activation**
  - [ ] `conv_blocks.py`: DownBlock/UpBlock에 `activation` 파라미터 추가, `nn.SiLU` 적용
  - [ ] `encoder.py`: `activation` 파라미터 추가, input_conv/bottleneck/DownBlock에 전달
  - [ ] `decoder.py`: `activation` 파라미터 추가, UpBlock에 전달
  - [ ] `tphasenet.py`: `activation` 파라미터 추가, from_config() 업데이트
  - [ ] `config/default.yaml`: `activation: "silu"` 추가
  - [ ] `config/defaults.py`: `"activation": "silu"` 추가
  - [ ] `pytest` 전수 통과 확인

- [ ] **Step 2: ResNet Block**
  - [ ] `conv_blocks.py` DownBlock: shortcut 추가, forward에 residual 적용
  - [ ] `conv_blocks.py` UpBlock: shortcut 추가, forward에 residual 적용
  - [ ] `pytest` 전수 통과 확인

- [ ] **Step 3: Skip Connection Attention**
  - [ ] `models/skip_attention.py`: SkipAttentionBlock 구현
  - [ ] `decoder.py`: skip_attention 파라미터 추가, SkipAttentionBlock 통합
  - [ ] `tphasenet.py`: skip_attention/lstm_hidden 파라미터 추가
  - [ ] `config/default.yaml`: `skip_attention`, `lstm_hidden` 추가
  - [ ] `config/defaults.py`: 동기화
  - [ ] `pytest` 전수 통과 확인

- [ ] **Step 4: Gap Insertion Augmentation**
  - [ ] `augmentation.py`: RandomGapInsertion 구현
  - [ ] `get_default_augmentation()`에 추가
  - [ ] `pytest` 전수 통과 확인
