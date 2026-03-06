# Seismic Picker 사용자 매뉴얼

## 목차

1. [개요](#1-개요)
2. [설치](#2-설치)
3. [추론 (Inference)](#3-추론-inference)
4. [학습 (Training)](#4-학습-training)
5. [설정 파일](#5-설정-파일)
6. [모델 아키텍처](#6-모델-아키텍처)
7. [데이터 파이프라인](#7-데이터-파이프라인)
8. [출력 형식](#8-출력-형식)
9. [Python API 사용법](#9-python-api-사용법)
10. [문제 해결](#10-문제-해결)

---

## 1. 개요

Seismic Picker는 **TPhaseNet** 기반의 지진파 P파/S파 자동 검출 시스템입니다.

### 핵심 개념

- **P파 (Primary wave)**: 지진 발생 시 가장 먼저 도착하는 종파
- **S파 (Secondary wave)**: P파 이후 도착하는 횡파
- **위상 검출 (Phase picking)**: 지진파형에서 P파와 S파의 정확한 도착 시각을 결정하는 작업

### 아키텍처

TPhaseNet은 PhaseNet(AUC 0.977)의 U-Net 구조에 Transformer 블록을 결합한 모델입니다.

- **입력**: `(B, 3, 6000)` — 3성분(Z, N, E) 지진파형, 60초 @ 100Hz
- **출력**: `(B, 3, 6000)` — 각 시점별 [Noise, P파, S파] 확률 (softmax)
- **파라미터**: 약 34M (NVIDIA 16-24GB GPU에서 학습/추론 가능)

### 두 가지 사용 모드

| 모드 | 설명 | 장점 |
|------|------|------|
| **SeisBench 모드** | 사전학습된 PhaseNet/EQTransformer 사용 | 즉시 사용 가능, 학습 불필요 |
| **TPhaseNet 모드** | STEAD로 자체 모델 학습 후 사용 | 커스터마이징 가능, 최신 아키텍처 |

---

## 2. 설치

### 2.1 시스템 요구사항

- **Python**: 3.9 이상
- **OS**: Linux, macOS, Windows
- **GPU**: NVIDIA GPU 권장 (16-24GB VRAM), CPU에서도 동작
- **디스크**: STEAD 학습 시 약 20GB 추가 필요

### 2.2 설치

```bash
# 저장소 클론
git clone https://github.com/mskim0210/seismic-picker.git
cd seismic-picker

# 의존성 설치
pip install -r requirements.txt
```

### 2.3 의존성 패키지

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `torch` | >= 2.0.0 | 딥러닝 프레임워크 |
| `numpy` | >= 1.24.0 | 수치 계산 |
| `scipy` | >= 1.10.0 | 신호 처리 (bandpass 필터, peak 검출) |
| `obspy` | >= 1.4.0 | 지진파형 I/O (mseed 포맷) |
| `h5py` | >= 3.8.0 | STEAD HDF5 데이터 로딩 |
| `pandas` | >= 2.0.0 | STEAD CSV 메타데이터 |
| `pyyaml` | >= 6.0 | 설정 파일 파싱 |
| `matplotlib` | >= 3.7.0 | 결과 시각화 |
| `tqdm` | >= 4.65.0 | 진행률 표시 |
| `seisbench` | >= 0.4.0 | 사전학습 모델 (선택 사항) |

### 2.4 GPU 설정 확인

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

디바이스는 자동으로 선택됩니다: CUDA → MPS (Apple Silicon) → CPU

---

## 3. 추론 (Inference)

### 3.1 SeisBench 사전학습 모델로 추론 (권장: 빠른 시작)

별도의 학습 없이 바로 사용할 수 있습니다.

```bash
# SeisBench 설치 (최초 1회)
pip install seisbench
```

#### PhaseNet 사용

```bash
python -m scripts.predict --input station.mseed --use-seisbench PhaseNet --pretrained stead
```

#### EQTransformer 사용

```bash
python -m scripts.predict --input station.mseed --use-seisbench EQTransformer --pretrained original
```

#### 사용 가능한 사전학습 가중치

| 가중치 | 학습 데이터 |
|--------|------------|
| `stead` | STEAD 데이터셋 (127만 지진파형) |
| `instance` | INSTANCE 이탈리아 지진 데이터셋 |
| `ethz` | 스위스 지진관측소 데이터 |
| `geofon` | GFZ GEOFON 글로벌 데이터 |
| `neic` | USGS NEIC 데이터 |
| `scedc` | 남부 캘리포니아 지진 데이터 |
| `original` | 모델 원저자의 가중치 |

가중치는 SeisBench가 자동으로 다운로드하여 캐시합니다.

### 3.2 TPhaseNet 모델로 추론

학습된 체크포인트 파일(`.pt`)이 필요합니다.

#### 단일 파일 추론

```bash
python -m scripts.predict --input station.mseed --model checkpoints/best_model.pt
```

#### 디렉토리 일괄 처리

```bash
python -m scripts.predict \
    --input-dir ./data/mseed/ \
    --model checkpoints/best_model.pt \
    --output-dir ./results/ \
    --format csv
```

#### 시각화

```bash
python -m scripts.predict --input station.mseed --model checkpoint.pt --plot
```

4개의 서브플롯이 표시됩니다:
1. **파형**: Z, N, E 3성분 지진파형
2. **P 확률**: P파 검출 확률 곡선 (파란색)
3. **S 확률**: S파 검출 확률 곡선 (빨간색)
4. **Noise 확률**: 배경 잡음 확률 곡선 (녹색)

검출된 pick은 각 서브플롯에 수직 점선으로 표시됩니다 (P: 파란색, S: 빨간색).

### 3.3 predict.py 전체 옵션

```
usage: predict.py [-h] (--input INPUT | --input-dir INPUT_DIR)
                  [--model MODEL] [--config CONFIG]
                  [--use-seisbench {PhaseNet,EQTransformer}]
                  [--pretrained PRETRAINED]
                  [--output OUTPUT] [--output-dir OUTPUT_DIR]
                  [--format {json,csv}]
                  [--device {cuda,cpu,mps}]
                  [--threshold THRESHOLD]
                  [--batch-size BATCH_SIZE]
                  [--plot]
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | 단일 mseed 파일 경로 | (필수, input-dir과 택 1) |
| `--input-dir` | mseed 디렉토리 경로 | (필수, input과 택 1) |
| `--model`, `-m` | 체크포인트 경로 (.pt) | None |
| `--config`, `-c` | 설정 YAML 경로 | 내장 기본값 |
| `--use-seisbench` | SeisBench 모델 선택 | None |
| `--pretrained` | 사전학습 가중치 이름 | `stead` |
| `--output`, `-o` | 출력 파일 경로 (단일 모드) | stdout |
| `--output-dir` | 출력 디렉토리 (배치 모드) | `{input-dir}/picks/` |
| `--format`, `-f` | 출력 포맷 | `json` |
| `--device`, `-d` | 연산 장치 | 자동 선택 |
| `--threshold`, `-t` | 최소 pick 확률 임계값 | `0.3` |
| `--batch-size`, `-b` | 배치 크기 | `1` |
| `--plot` | 결과 시각화 | 비활성 |

---

## 4. 학습 (Training)

### 4.1 STEAD 데이터셋 준비

STEAD (STanford EArthquake Dataset)는 127만 개의 3성분 지진파형을 포함하는 대규모 데이터셋입니다.

#### 다운로드 안내 확인

```bash
python -m scripts.download_stead --method manual
```

#### 필요한 파일

| 파일 | 크기 | 내용 |
|------|------|------|
| `merged.hdf5` | ~17GB | 파형 데이터 (1,265,657 traces, 60초 @ 100Hz) |
| `merged.csv` | ~250MB | 메타데이터 (P/S arrival time, 진원지 정보 등) |

다운로드 방법:
1. [STEAD GitHub](https://github.com/smousavi05/STEAD) README의 다운로드 링크 사용
2. 또는 DOI로 접근: https://doi.org/10.1109/ACCESS.2019.2947848

#### SeisBench를 통한 다운로드 (대안)

```bash
python -m scripts.download_stead --method seisbench --output-dir ./data/stead
```

### 4.2 학습 실행

#### 소량 데이터로 테스트 (권장: 처음 실행 시)

```bash
python -m scripts.train \
    --csv /path/to/merged.csv \
    --hdf5 /path/to/merged.hdf5 \
    --max-samples 1000 \
    --epochs 5
```

#### 전체 데이터 학습

```bash
python -m scripts.train \
    --csv /path/to/merged.csv \
    --hdf5 /path/to/merged.hdf5 \
    --config config/default.yaml \
    --checkpoint-dir ./checkpoints
```

#### 학습 이어하기 (resume)

```bash
python -m scripts.train \
    --csv /path/to/merged.csv \
    --hdf5 /path/to/merged.hdf5 \
    --resume checkpoints/checkpoint_epoch50.pt
```

### 4.3 train.py 전체 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--csv` | STEAD merged.csv 경로 | (필수) |
| `--hdf5` | STEAD merged.hdf5 경로 | (필수) |
| `--config`, `-c` | 설정 YAML 경로 | 내장 기본값 |
| `--epochs` | 최대 에폭 수 | `100` |
| `--batch-size` | 배치 크기 | `64` |
| `--lr` | 학습률 | `1e-3` |
| `--device` | 연산 장치 (`cuda`/`cpu`/`mps`) | 자동 선택 |
| `--num-workers` | DataLoader 워커 수 | `4` |
| `--checkpoint-dir` | 체크포인트 저장 디렉토리 | `./checkpoints` |
| `--resume` | 이어서 학습할 체크포인트 경로 | None |
| `--max-samples` | 최대 샘플 수 제한 (디버깅용) | 전체 |
| `--no-augment` | 데이터 증강 비활성화 | 활성 |

### 4.4 학습 과정

학습 중 아래 정보가 에폭마다 출력됩니다:

```
Epoch   1/100 | Train Loss: 0.0523 | Val Loss: 0.0412 | P-F1: 0.621 | S-F1: 0.534 | LR: 1.00e-03 | Time: 245s
  -> Best model saved (val_loss: 0.0412)
Epoch   2/100 | Train Loss: 0.0387 | Val Loss: 0.0356 | P-F1: 0.714 | S-F1: 0.652 | LR: 1.00e-03 | Time: 243s
  -> Best model saved (val_loss: 0.0356)
...
```

#### 데이터 분할

STEAD 데이터는 `source_id` 해시 기반으로 자동 분할됩니다:

| 분할 | 비율 | 용도 |
|------|------|------|
| Train | 80% | 모델 학습 |
| Validation | 10% | 학습 중 성능 모니터링, early stopping |
| Test | 10% | 최종 평가 |

#### 체크포인트

- `best_model.pt`: 검증 손실이 가장 낮은 모델 (자동 저장)
- `checkpoint_epoch{N}.pt`: 10 에폭마다 주기적 저장

체크포인트에 저장되는 내용:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "val_loss": float,
    "best_val_loss": float,
    "metrics": {"p_f1": ..., "s_f1": ..., ...},
    "config": dict,
}
```

#### Early Stopping

검증 손실이 15 에폭 연속으로 개선되지 않으면 학습이 자동 중단됩니다.

### 4.5 손실함수

두 가지 손실함수를 지원합니다:

#### Weighted Cross-Entropy (기본)

```
loss = -sum_c(w_c * y_c * log(p_c + eps))
```

- 클래스 가중치: Noise=1.0, P=30.0, S=30.0
- P/S 도착 구간에 높은 가중치를 부여하여 클래스 불균형 해소

#### Focal Cross-Entropy

```
FL = -alpha * (1 - p)^gamma * y * log(p + eps)
```

- alpha: [0.1, 0.45, 0.45], gamma: 2.0
- 쉬운 샘플(noise)의 가중치를 줄여 어려운 샘플(P/S)에 집중

설정에서 변경:
```yaml
training:
  loss: "weighted_ce"  # 또는 "focal"
```

### 4.6 학습률 스케줄러

| 스케줄러 | 설명 |
|----------|------|
| `reduce_on_plateau` (기본) | 검증 손실이 5 에폭간 개선 없으면 LR × 0.5 |
| `one_cycle` | OneCycleLR - 코사인 어닐링, 전체 학습의 30%에서 max LR 도달 |

### 4.7 데이터 증강

학습 시 기본적으로 5가지 데이터 증강이 적용됩니다:

| 증강 | 확률 | 파라미터 | 설명 |
|------|------|----------|------|
| Gaussian Noise | 50% | SNR 5-30 dB | 배경 잡음 추가 |
| Amplitude Scale | 50% | 0.5-2.0x | 진폭 랜덤 스케일링 |
| Time Shift | 30% | ±200 samples | 시간축 이동 (라벨도 함께) |
| Channel Drop | 10% | 1 채널 | 결측 성분 시뮬레이션 |
| Polarity Flip | 50% | 전체 반전 | 극성 반전 |

`--no-augment` 옵션으로 비활성화할 수 있습니다.

---

## 5. 설정 파일

### 5.1 기본 설정 (`config/default.yaml`)

```yaml
model:
  name: "TPhaseNet"
  in_channels: 3          # 입력 채널 (Z, N, E)
  classes: 3              # 출력 클래스 (Noise, P, S)
  filters_root: 8         # 기본 필터 수 (레벨별 2배씩 증가)
  depth: 8                # U-Net 깊이 (8 레벨)
  kernel_size: 7          # 컨볼루션 커널 크기
  stride: 4               # 다운샘플링/업샘플링 스트라이드
  transformer_start_level: 4  # Transformer 적용 시작 레벨
  n_heads: 4              # Multi-head attention 헤드 수
  ff_dim_factor: 4        # Feed-forward 차원 확장 배수
  dropout: 0.1            # Dropout 비율

data:
  target_length: 6000     # 입력 길이 (60초 @ 100Hz)
  sampling_rate: 100.0    # 샘플링 레이트 (Hz)
  filter:
    enabled: true
    freq_min: 0.5         # 밴드패스 최저 주파수 (Hz)
    freq_max: 45.0        # 밴드패스 최고 주파수 (Hz)
    corners: 4            # Butterworth 필터 차수
  normalize:
    method: "std"          # 정규화 방법 (std, peak, minmax)
    epsilon: 1.0e-8

inference:
  device: "cuda"
  peak_detection:
    min_height: 0.3       # 최소 확률 임계값
    min_distance: 100     # 피크 간 최소 거리 (샘플, 1초)
    min_prominence: 0.1   # 최소 prominence
  sliding_window:
    window_size: 6000     # 슬라이딩 윈도우 크기 (샘플)
    step: 3000            # 윈도우 이동 간격 (50% 오버랩)
  output_format: "json"
```

### 5.2 학습 설정 (기본값, YAML에 추가 가능)

```yaml
training:
  batch_size: 64
  max_epochs: 100
  loss: "weighted_ce"            # "weighted_ce" 또는 "focal"
  class_weights: [1.0, 30.0, 30.0]
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.00001
    betas: [0.9, 0.999]
  scheduler:
    type: "reduce_on_plateau"    # "reduce_on_plateau" 또는 "one_cycle"
    factor: 0.5
    patience: 5
    min_lr: 0.000001
  gradient_clip: 1.0
  early_stopping:
    patience: 15
  mixed_precision: true          # CUDA AMP (자동 혼합 정밀도)
  checkpoint_dir: "./checkpoints"
```

### 5.3 설정 우선순위

CLI 인수 > YAML 설정 파일 > 내장 기본값

예시: YAML에 `batch_size: 64`로 설정되어 있어도 `--batch-size 32`를 지정하면 32가 적용됩니다.

---

## 6. 모델 아키텍처

### 6.1 TPhaseNet 구조

```
입력: (B, 3, 6000) — 3성분 지진파형

Encoder (7-level 다운샘플링)
├── Input Conv: 3 → 8 채널
├── Level 1: DownBlock (8 → 8)
├── Level 2: DownBlock (8 → 16)
├── Level 3: DownBlock (16 → 32)
├── Level 4: DownBlock (32 → 64) + Transformer
├── Level 5: DownBlock (64 → 128) + Transformer
├── Level 6: DownBlock (128 → 256) + Transformer
├── Level 7: DownBlock (256 → 512) + Transformer
└── Bottleneck: Transformer

Decoder (7-level 업샘플링)
├── Level 7: UpBlock (512 → 256) + Skip Connection
├── Level 6: UpBlock (256 → 128) + Skip Connection
├── Level 5: UpBlock (128 → 64) + Skip Connection
├── Level 4: UpBlock (64 → 32) + Skip Connection
├── Level 3: UpBlock (32 → 16) + Skip Connection
├── Level 2: UpBlock (16 → 8) + Skip Connection
├── Level 1: UpBlock (8 → 8) + Skip Connection
└── Output: 1x1 Conv → Softmax (3 classes)

출력: (B, 3, 6000) — [Noise, P, S] 확률
```

### 6.2 핵심 블록

#### DownBlock
```
Input → Conv1d → BatchNorm → ReLU → Conv1d(stride=4) → BatchNorm → ReLU
                                                                      ↓
                                                              (skip, downsampled)
```

#### UpBlock
```
Input → ConvTranspose1d → BatchNorm → ReLU → Concat(skip) → Conv1d → BatchNorm → ReLU
```

#### TransformerBlock
- Sinusoidal Positional Encoding
- Multi-head Self-Attention (4 heads)
- Feed-Forward Network (GELU 활성화)
- Residual Connection + LayerNorm
- Level 4 이상의 인코더에만 적용 (저해상도 특징에서 전역 관계 학습)

### 6.3 라벨 생성

P파/S파 도착 시각을 중심으로 Gaussian 곡선을 생성합니다:

```
P_label(t) = exp(-(t - t_P)^2 / (2 * sigma^2))    sigma = 20 (0.2초)
S_label(t) = exp(-(t - t_S)^2 / (2 * sigma^2))
Noise(t) = clip(1.0 - P(t) - S(t), 0, 1)
```

---

## 7. 데이터 파이프라인

### 7.1 전처리 순서

mseed 파일 로드 시 다음 순서로 전처리됩니다:

1. **채널 정렬**: Z, N, E 순서로 재배치
2. **리샘플링**: 목표 샘플링 레이트(100Hz)로 변환
3. **Demean**: 채널별 평균 제거
4. **Detrend**: 선형 트렌드 제거
5. **Bandpass Filter**: 0.5-45Hz Butterworth 필터 (4차)
6. **Normalize**: 표준편차로 정규화

### 7.2 mseed 파일 요구사항

- **포맷**: miniSEED (ObsPy 호환)
- **채널**: 3성분 (Z, N, E) — 누락 시 0으로 채움
- **샘플링 레이트**: 자동 리샘플링 (100Hz로 변환)
- **길이 제한 없음**: 짧은 파형은 zero-padding, 긴 파형은 슬라이딩 윈도우 처리

### 7.3 STEAD 데이터셋

- **크기**: 1,265,657 traces (3성분, 60초, 100Hz)
- **채널 순서**: HDF5 원본 [E, N, Z] → 모델 입력 [Z, N, E]로 자동 변환
- **분할**: source_id 해시 기반 (train 80%, val 10%, test 10%)
- **멀티프로세싱**: worker별 독립적 HDF5 핸들 사용 (안정성 보장)

---

## 8. 출력 형식

### 8.1 JSON 출력 (기본)

```json
{
  "station": "KMA01",
  "network": "KS",
  "location": "",
  "start_time": "2024-01-15T03:22:10.000000Z",
  "channels": ["HHZ", "HHN", "HHE"],
  "picks": [
    {
      "phase": "P",
      "time": "2024-01-15T03:22:25.340000Z",
      "confidence": 0.9523,
      "uncertainty_sec": 0.08,
      "sample_index": 1534
    },
    {
      "phase": "S",
      "time": "2024-01-15T03:22:31.780000Z",
      "confidence": 0.8917,
      "uncertainty_sec": 0.12,
      "sample_index": 2178
    }
  ]
}
```

각 pick의 필드:

| 필드 | 설명 |
|------|------|
| `phase` | 위상 종류 (`P` 또는 `S`) |
| `time` | 절대 도착 시각 (UTC) |
| `confidence` | 검출 신뢰도 (0.0 ~ 1.0) |
| `uncertainty_sec` | 시간 불확실성 (초) |
| `sample_index` | 파형 내 샘플 인덱스 |

### 8.2 CSV 출력

디렉토리 일괄 처리 시 `all_picks.csv` 파일이 생성됩니다:

```csv
network,station,location,phase,time,confidence,uncertainty_sec
KS,KMA01,,P,2024-01-15T03:22:25.340000Z,0.9523,0.08
KS,KMA01,,S,2024-01-15T03:22:31.780000Z,0.8917,0.12
KS,KMA02,,P,2024-01-15T03:22:26.120000Z,0.9234,0.10
...
```

### 8.3 Pick 검출 파라미터

`scipy.signal.find_peaks`를 사용하여 확률 곡선에서 피크를 검출합니다:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `min_height` | 0.3 | 최소 확률 임계값. 낮추면 더 많은 pick 검출 (오검출 증가) |
| `min_distance` | 100 | 피크 간 최소 거리 (100 샘플 = 1초). 인접 중복 pick 방지 |
| `min_prominence` | 0.1 | 주변 대비 최소 돌출도. 낮으면 약한 피크도 검출 |

#### 물리적 제약

S파 도착은 반드시 P파 이후여야 합니다. 이 제약에 위배되는 pick은 자동 제거됩니다.

### 8.4 슬라이딩 윈도우

60초보다 긴 파형은 슬라이딩 윈도우로 처리됩니다:

- **윈도우 크기**: 6000 샘플 (60초)
- **이동 간격**: 3000 샘플 (30초, 50% 오버랩)
- **병합**: 겹치는 구간의 확률값을 평균

---

## 9. Python API 사용법

### 9.1 TPhaseNet 모델 직접 사용

```python
from models.tphasenet import TPhaseNet
import torch

# 모델 생성
model = TPhaseNet(
    in_channels=3,
    classes=3,
    filters_root=8,
    depth=8,
    kernel_size=7,
    stride=4,
    transformer_start_level=4,
    n_heads=4,
    ff_dim_factor=4,
    dropout=0.1,
)
print(f"Parameters: {model.count_parameters():,}")

# 추론
x = torch.randn(1, 3, 6000)  # (batch, channels, samples)
with torch.no_grad():
    probs = model(x)  # (1, 3, 6000) [Noise, P, S]
print(f"Output shape: {probs.shape}")
print(f"Softmax sum: {probs[0, :, 0].sum():.4f}")  # = 1.0
```

### 9.2 SeismicPicker (End-to-End 추론)

```python
from inference.picker import SeismicPicker

# 초기화
picker = SeismicPicker(
    model_path="checkpoints/best_model.pt",
    config_path="config/default.yaml",  # 생략 시 기본값
    device="cuda",  # 생략 시 자동 선택
)

# 단일 파일 추론
result = picker.pick("station.mseed")
print(result)

# 일괄 처리
results = picker.pick_batch(
    ["file1.mseed", "file2.mseed"],
    output_dir="./results/",
    output_format="json",
)

# 확률 곡선 얻기 (시각화용)
prob_curves, waveform, metadata = picker.get_probabilities("station.mseed")
# prob_curves: (3, N) — [Noise, P, S]
# waveform: (3, N) — 전처리된 파형
```

### 9.3 SeisBench Picker

```python
from inference.seisbench_picker import SeisBenchPicker

# 초기화
picker = SeisBenchPicker(
    model_name="PhaseNet",     # "PhaseNet" 또는 "EQTransformer"
    pretrained="stead",        # 사전학습 가중치
    device="cpu",
    p_threshold=0.3,
    s_threshold=0.3,
)

# 추론
result = picker.pick("station.mseed")

# 사용 가능한 모델 목록 확인
SeisBenchPicker.list_models()
```

### 9.4 데이터 전처리

```python
from data.preprocessing import preprocess
import numpy as np

# (3, N) 형태의 파형
waveform = np.random.randn(3, 6000).astype(np.float32)

# 전처리 적용
processed = preprocess(
    waveform,
    sampling_rate=100.0,
    freq_min=0.5,
    freq_max=45.0,
    normalize_method="std",
)
```

### 9.5 mseed 파일 로드

```python
from data.mseed_loader import load_mseed, load_mseed_stream

# 고정 길이 로드 (6000 샘플로 자동 패딩/트림)
waveform, metadata = load_mseed("station.mseed", target_length=6000)

# 가변 길이 로드 (슬라이딩 윈도우용)
waveform, metadata = load_mseed_stream("station.mseed", target_sampling_rate=100.0)
```

---

## 10. 문제 해결

### GPU 메모리 부족 (CUDA Out of Memory)

```bash
# 배치 크기 줄이기
python -m scripts.train --csv merged.csv --hdf5 merged.hdf5 --batch-size 32

# 추론 시 CPU 사용
python -m scripts.predict --input station.mseed --model checkpoint.pt --device cpu
```

### SeisBench 설치 오류

```bash
# seisbench 의존성 먼저 설치
pip install obspy
pip install seisbench
```

### mseed 파일 읽기 오류

- 파일이 유효한 miniSEED 형식인지 확인
- ObsPy로 직접 읽어보기: `from obspy import read; st = read("file.mseed"); print(st)`
- 3성분이 아닌 경우에도 동작합니다 (누락 채널은 0으로 채움)

### 학습이 수렴하지 않는 경우

1. 소량 데이터로 오버피팅 테스트: `--max-samples 100 --epochs 50`
2. 학습률 조정: `--lr 1e-4` 또는 `--lr 5e-4`
3. 데이터 증강 비활성화하여 기본 성능 확인: `--no-augment`
4. Focal Loss로 변경: 설정에서 `loss: "focal"`

### pick 결과가 너무 많거나 적은 경우

- **pick이 너무 많을 때**: threshold 높이기 → `--threshold 0.5` 또는 `0.7`
- **pick이 너무 적을 때**: threshold 낮추기 → `--threshold 0.1` 또는 `0.2`
- 설정 파일에서 `min_prominence`도 조정 가능

### 학습 평가 메트릭

| 메트릭 | 설명 | 목표 |
|--------|------|------|
| P-F1 | P파 검출 F1 스코어 | > 0.9 |
| S-F1 | S파 검출 F1 스코어 | > 0.85 |
| p_mean_residual_sec | P파 도착 시각 오차 평균 (초) | < 0.1 |
| s_mean_residual_sec | S파 도착 시각 오차 평균 (초) | < 0.15 |

메트릭 계산 기준: 예측과 실제 도착 시각의 차이가 0.5초 이내이면 True Positive으로 판정합니다.
