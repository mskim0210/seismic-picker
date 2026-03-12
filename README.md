# Seismic Picker

**TPhaseNet** 기반 지진파 P파/S파 자동 검출 시스템

TPhaseNet은 PhaseNet(U-Net)에 Transformer 블록을 결합한 딥러닝 모델로, 3성분 지진파형에서 P파와 S파 도착 시각을 자동으로 검출합니다.

## 성능

STEAD 테스트셋 (127,652 traces) 기준, threshold=0.8:

| Phase | Precision | Recall | F1 | Residual (sec) |
|-------|-----------|--------|------|----------------|
| P-wave | 0.9708 | 0.9951 | **0.9828** | -0.007 ± 0.039 |
| S-wave | 0.9569 | 0.9738 | **0.9653** | +0.004 ± 0.111 |

- 모델 파라미터: **517,691** (0.5M)
- 학습: STEAD 전체 (1,011,324 train / 126,681 val), 48 epochs
- GPU: NVIDIA RTX 5090

## 주요 특징

- **TPhaseNet 아키텍처**: U-Net + Transformer (약 517K 파라미터)
- **입출력**: `(B, 3, 6000)` → `(B, 3, 6000)` [Noise, P파, S파] 확률 (softmax)
- **SeisBench 연동**: PhaseNet/EQTransformer 사전학습 가중치로 즉시 추론 가능
- **STEAD 학습**: 127만 지진파형 데이터셋으로 자체 모델 학습 지원
- **가변 길이 지원**: 슬라이딩 윈도우 추론으로 긴 파형도 처리
- **시각화**: 파형 + 확률 곡선 + pick 마커 플롯

## 프로젝트 구조

```
seismic-picker/
├── models/                  # 딥러닝 모델
│   ├── tphasenet.py         # Encoder+Decoder 조립, from_config() 지원
│   ├── encoder.py           # 4-level 인코더 (Level 3-4에 Transformer)
│   ├── decoder.py           # 4-level 디코더 + skip connection + softmax 출력
│   ├── conv_blocks.py       # DownBlock/UpBlock (Conv1d+BN+ReLU)
│   └── transformer_block.py # Sinusoidal PE + Transformer Encoder Layer
├── data/                    # 데이터 로딩 및 전처리
│   ├── stead_dataset.py     # STEAD HDF5 PyTorch Dataset (md5 해시 기반 split)
│   ├── preprocessing.py     # demean, detrend, bandpass(0.5-45Hz), normalize
│   ├── augmentation.py      # 노이즈, 스케일링, 시간이동, 채널드롭, 극성반전
│   ├── mseed_loader.py      # ObsPy로 mseed 읽기, 3성분 정렬, 전처리
│   └── label_utils.py       # Gaussian 라벨 생성 (sigma=20, 0.2초)
├── training/                # 학습 파이프라인
│   ├── trainer.py           # 학습루프, 검증, 체크포인트, early stopping, AMP
│   ├── losses.py            # Weighted CE (1:30:30) + Focal CE
│   └── metrics.py           # P/S precision, recall, F1, pick residual
├── inference/               # 추론 파이프라인
│   ├── picker.py            # End-to-end 추론 (mseed→pick), 가변길이 지원
│   ├── seisbench_picker.py  # SeisBench PhaseNet/EQTransformer 사전학습 모델 래퍼
│   ├── postprocessing.py    # find_peaks로 P/S pick 추출, 슬라이딩 윈도우 병합
│   └── output_formatter.py  # JSON/CSV 출력, 절대 시각 변환
├── scripts/                 # CLI 스크립트
│   ├── predict.py           # 추론 CLI (단일파일/디렉토리, 시각화 옵션)
│   ├── train.py             # 학습 CLI (resume, max-samples 지원)
│   ├── evaluate.py          # 모델 평가 (threshold sweep, residual 분석)
│   ├── benchmark.py         # 모델 간 벤치마크 비교 (TPhaseNet vs SeisBench)
│   └── download_stead.py    # STEAD 다운로드 안내/SeisBench 연동
├── config/
│   ├── default.yaml         # 기본 설정 (모델, 데이터, 추론)
│   └── defaults.py          # 기본 설정값 단일 소스 (모든 스크립트에서 참조)
├── tests/                   # 단위 테스트 (82개)
├── requirements.txt
└── setup.py
```

## 설치

### 요구사항

- Python >= 3.9
- NVIDIA GPU (8GB+ VRAM 권장, CPU도 가능)

### 설치 방법

```bash
git clone https://github.com/mskim0210/seismic-picker.git
cd seismic-picker
pip install -r requirements.txt
```

## 빠른 시작

### 방법 1: SeisBench 사전학습 모델 (즉시 사용 가능)

별도의 학습 없이 사전학습된 가중치로 바로 추론할 수 있습니다.

```bash
pip install seisbench

# PhaseNet (STEAD 학습 가중치)으로 바로 추론
python -m scripts.predict --input station.mseed --use-seisbench PhaseNet --pretrained stead

# EQTransformer (원본 가중치)로 추론
python -m scripts.predict --input station.mseed --use-seisbench EQTransformer --pretrained original
```

사용 가능한 가중치: `stead`, `instance`, `ethz`, `geofon`, `neic`, `scedc`, `original` 등

### 방법 2: TPhaseNet 자체 모델 학습

```bash
# 1. STEAD 다운로드 안내
python -m scripts.download_stead --method manual

# 2. 소량 테스트 학습
python -m scripts.train --csv merged.csv --hdf5 merged.hdf5 --max-samples 1000 --epochs 5

# 3. 전체 학습
python -m scripts.train --csv merged.csv --hdf5 merged.hdf5 --config config/default.yaml

# 4. 학습된 모델로 추론
python -m scripts.predict --input station.mseed --model checkpoints/best_model.pt
```

### 추론 예시

```bash
# 단일 mseed 파일 추론
python -m scripts.predict --input station.mseed --model checkpoint.pt

# 디렉토리 일괄 처리 + CSV 출력
python -m scripts.predict --input-dir ./data/ --model checkpoint.pt --output-dir ./results/ --format csv

# 시각화 (파형 + 확률 곡선 + pick 마커)
python -m scripts.predict --input station.mseed --model checkpoint.pt --plot
```

### 평가

```bash
# 기본 평가
python -m scripts.evaluate \
    --model checkpoints/best_model.pt \
    --csv /path/to/merged.csv \
    --hdf5 /path/to/merged.hdf5

# threshold 범위 분석
python -m scripts.evaluate \
    --model checkpoints/best_model.pt \
    --csv /path/to/merged.csv \
    --hdf5 /path/to/merged.hdf5 \
    --sweep-thresholds

# 모델 간 벤치마크 비교
python -m scripts.benchmark \
    --model checkpoints/best_model.pt \
    --csv /path/to/merged.csv \
    --hdf5 /path/to/merged.hdf5
```

## 설정

`config/default.yaml`에서 모델, 데이터 전처리, 추론 파라미터를 설정할 수 있습니다.

```yaml
model:
  filters_root: 8       # 기본 채널 수
  depth: 5              # U-Net 깊이 (5 레벨)
  kernel_size: 7        # 컨볼루션 커널 크기
  stride: 2             # 다운샘플링 스트라이드
  transformer_start_level: 3  # Transformer 적용 시작 레벨
  n_heads: 4            # 어텐션 헤드 수
  dropout: 0.1

data:
  target_length: 6000   # 60초 @ 100 Hz
  sampling_rate: 100.0
  filter:
    freq_min: 0.5       # 밴드패스 최저 주파수 (Hz)
    freq_max: 45.0      # 밴드패스 최고 주파수 (Hz)

inference:
  peak_detection:
    min_height: 0.3     # 최소 확률 임계값
    min_distance: 100   # 피크 간 최소 거리 (샘플)
    min_prominence: 0.1 # 최소 prominence
  sliding_window:
    window_size: 6000   # 슬라이딩 윈도우 크기
    step: 3000          # 윈도우 이동 간격 (50% 오버랩)
```

## 의존성

| 패키지 | 용도 |
|--------|------|
| torch >= 2.0.0 | 딥러닝 프레임워크 |
| obspy >= 1.4.0 | 지진파형 I/O (mseed) |
| numpy, scipy | 수치 계산, 신호 처리 |
| h5py, pandas | STEAD 데이터셋 로딩 |
| seisbench >= 0.4.0 | 사전학습 모델 (선택) |
| matplotlib | 시각화 |
| pyyaml | 설정 파일 |
| tqdm | 진행률 표시 |

## 참고

- [PhaseNet](https://doi.org/10.1093/gji/ggy423) - Zhu & Beroza, 2019
- [TPhaseNet](https://doi.org/10.1785/0220230402) - 2024
- [STEAD](https://doi.org/10.1109/ACCESS.2019.2947848) - Mousavi et al., 2019
- [SeisBench](https://github.com/seisbench/seisbench) - Woollam et al., 2022

## 라이선스

MIT License
