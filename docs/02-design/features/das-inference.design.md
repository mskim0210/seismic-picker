# DAS Inference Design Document

> Plan Reference: `docs/01-plan/features/das-inference.plan.md`

---

## 1. Overview

기존 TPhaseNet 모델(STEAD 학습, 100Hz 3성분)을 1000Hz 단일성분 DAS(TDMS) 데이터에 적용하기 위한 설계.
기존 파이프라인을 최대한 재활용하되, DAS 데이터의 특성(1000Hz, 단일성분, TDMS 포맷)에 맞는 로더만 신규 작성.

---

## 2. Architecture

### 2.1 System Context

```
                    ┌─────────────────────────────────┐
                    │         scripts/predict.py       │
                    │  (.tdms 확장자 감지 → 분기)       │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │      inference/picker.py         │
                    │  pick_tdms() / get_prob_tdms()   │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼─────┐  ┌──────▼──────┐  ┌──────▼──────────┐
   │ data/           │  │ models/      │  │ inference/       │
   │ tdms_loader.py  │  │ tphasenet.py │  │ postprocessing.py│
   │ (NEW)           │  │ (no change)  │  │ (no change)      │
   └────────────────┘  └─────────────┘  └─────────────────┘
```

### 2.2 Data Flow (Detailed)

```
Step 1: TDMS Read
  nptdms.TdmsFile(path) → group["Measurement"] → channel[idx]
  → raw: np.ndarray (int16, 30000 samples)

Step 2: Type Conversion
  int16 → float64

Step 3: Downsample
  scipy.signal.decimate(data, factor=10, zero_phase=True)
  → 30000 samples → 3000 samples (1000Hz → 100Hz)

Step 4: 3-Channel Replication
  (3000,) → np.stack([data, data, data]) → (3, 3000)

Step 5: Preprocessing (reuse existing)
  preprocessing.preprocess(data, sampling_rate=100.0, config)
  → demean → detrend → bandpass(0.5-45Hz) → normalize
  → (3, 3000) float32

Step 6: Zero-Pad
  picker._infer_single(waveform, target_length=6000)
  → (3, 3000) → pad → (3, 6000) → model → (3, 6000) → trim → (3, 3000)

Step 7: Pick Extraction (reuse existing)
  postprocessing.extract_picks(prob_curves, ...)
  → list of {phase, sample_index, time_offset_sec, confidence}

Step 8: Visualization (reuse existing plot_results pattern)
  DAS 파형 (1ch) + P확률 + S확률 + Noise확률 + pick 마커
```

---

## 3. Component Design

### 3.1 `data/tdms_loader.py` (NEW)

```python
"""DAS TDMS 파일 로더.

1000Hz 단일성분 DAS 데이터를 TPhaseNet 입력 형태(3ch, 100Hz)로 변환.
"""

import numpy as np
from scipy.signal import decimate
from .preprocessing import preprocess


def load_tdms_channel(
    tdms_path: str,
    channel_index: int = 0,
    target_sampling_rate: float = 100.0,
    config: dict = None,
) -> tuple:
    """TDMS 파일에서 단일 DAS 채널을 로드하여 모델 입력 형태로 변환.

    Args:
        tdms_path: TDMS 파일 경로
        channel_index: DAS 채널 인덱스 (0-based, 기본 0)
        target_sampling_rate: 목표 샘플링 레이트 (Hz), 기본 100.0
        config: data 설정 dict (전처리 파라미터)

    Returns:
        waveform: (3, N) numpy array (float32) — 3채널 복제, 다운샘플링 후
        metadata: dict — sampling_rate, n_samples, channel_index, ...
    """
```

**내부 구현 상세:**

1. **TDMS 읽기**: `nptdms.TdmsFile(tdms_path)` 사용
   - group name: `"Measurement"` (DAS 표준)
   - channel: `group.channels()[channel_index]`
   - raw data: `channel.data` → `np.ndarray` (int16)

2. **메타데이터 추출**:
   ```python
   metadata = {
       "source_file": str(tdms_path),
       "channel_index": channel_index,
       "channel_name": channel.name,
       "original_sampling_rate": float(channel.properties.get("SamplingFrequency", 1000)),
       "spatial_resolution": float(channel.properties.get("SpatialResolution", 0)),
       "start_position": float(channel.properties.get("StartPosition", 0)),
       "n_original_samples": len(raw_data),
       "sampling_rate": target_sampling_rate,
       "data_type": "DAS",
   }
   ```

3. **다운샘플링**: `scipy.signal.decimate(data, factor, zero_phase=True)`
   - `factor = int(original_rate / target_sampling_rate)` → 10
   - `zero_phase=True`로 anti-aliasing 적용
   - 정수 비율이 아닌 경우 → `ValueError` raise

4. **3채널 복제**: `np.stack([data, data, data], axis=0)` → `(3, N)`

5. **전처리**: 기존 `preprocess()` 호출
   - `preprocess(waveform, sampling_rate=100.0, config=config)`

**에러 처리:**
- `nptdms` 미설치 → `ImportError` with 설치 안내 메시지
- 채널 인덱스 범위 초과 → `IndexError` with 유효 범위 안내
- TDMS 읽기 실패 → 원본 예외 전달

### 3.2 `inference/picker.py` (MODIFY)

기존 `SeismicPicker` 클래스에 2개 메서드 추가:

```python
def pick_tdms(self, tdms_path, channel_index=0):
    """TDMS DAS 파일에서 위상 pick 추출.

    Args:
        tdms_path: TDMS 파일 경로
        channel_index: DAS 채널 인덱스

    Returns:
        dict: {
            'picks': list of pick dicts,
            'channel_index': int,
            'data_type': 'DAS',
            ...metadata
        }
    """
```

**구현 로직:**
1. `load_tdms_channel(tdms_path, channel_index, self.sampling_rate, data_cfg)` 호출
2. `self._infer_single(waveform, self.target_length)` 호출
3. `extract_picks(prob_curves, ...)` 호출
4. DAS는 절대 시각이 없으므로 `format_picks_absolute()` 대신 상대 시각 반환
5. metadata와 picks를 dict로 반환

```python
def get_probabilities_tdms(self, tdms_path, channel_index=0):
    """TDMS DAS 확률 곡선 반환 (시각화용).

    Returns:
        prob_curves: (3, N) numpy array [Noise, P, S]
        waveform: (3, N) numpy array
        metadata: dict
    """
```

**구현 로직:**
1. `load_tdms_channel()` 호출
2. `self._infer_single()` 호출
3. `(prob_curves, waveform, metadata)` 튜플 반환

### 3.3 `scripts/predict.py` (MODIFY)

최소 수정: `--input` 파일의 확장자가 `.tdms`인 경우 DAS 분기.

**변경 사항:**

1. CLI 인자 추가:
   ```python
   parser.add_argument(
       "--channel", type=int, default=0,
       help="DAS 채널 인덱스 (TDMS 파일 전용, 기본: 0)"
   )
   ```

2. `main()` 내 분기 로직:
   ```python
   if args.input:
       is_tdms = args.input.lower().endswith(".tdms")

       if is_tdms and use_seisbench:
           print("Error: TDMS 파일은 TPhaseNet 모델만 지원합니다.")
           sys.exit(1)

       if is_tdms:
           # DAS 모드
           if args.plot:
               prob_curves, waveform, metadata = picker.get_probabilities_tdms(
                   args.input, channel_index=args.channel
               )
               picks = extract_picks(prob_curves, ...)
               plot_results_das(waveform, prob_curves, picks, metadata, ...)
               result = {"picks": picks, **metadata}
           else:
               result = picker.pick_tdms(args.input, channel_index=args.channel)
       else:
           # 기존 mseed 모드 (변경 없음)
           ...
   ```

3. `plot_results_das()` 함수 추가 (기존 `plot_results()` 변형):
   - 3성분 대신 DAS 단일 파형 1개만 표시
   - 타이틀에 채널 인덱스 표시
   - 나머지(P/S/Noise 확률, pick 마커)는 동일

---

## 4. `plot_results_das()` Design

기존 `plot_results()`의 DAS 버전. 차이점:

| 항목 | plot_results (mseed) | plot_results_das (DAS) |
|------|---------------------|----------------------|
| 파형 패널 | 3성분 (Z, N, E) | 1채널 (DAS Ch.N) |
| 제목 | Station: XX \| Start: ... | DAS Ch.500 \| File: ... |
| 서브플롯 수 | 4 (3ch + P + S + Noise) | 4 (1ch + P + S + Noise) |

```python
def plot_results_das(waveform, prob_curves, picks, metadata, sampling_rate=100.0):
    """DAS 파형과 확률 곡선, pick 결과를 시각화."""
    # axes[0]: DAS 단일 채널 파형 (waveform[0] 사용)
    # axes[1]: P 확률
    # axes[2]: S 확률
    # axes[3]: Noise 확률
    # pick 마커: 모든 축에 axvline
```

---

## 5. Dependencies

| 패키지 | 버전 | 용도 | 설치 방법 |
|--------|------|------|----------|
| `nptdms` | >= 1.0.0 | TDMS 파일 I/O | `conda install -c conda-forge nptdms` |
| `scipy` | (이미 설치) | `decimate()` | - |

**`requirements.txt` 변경**: `nptdms`는 optional dependency (DAS 사용 시만 필요).
→ `setup.py`의 `extras_require`에 추가:

```python
extras_require={
    "seisbench": ["seisbench>=0.4.0"],
    "das": ["nptdms>=1.0.0"],
}
```

---

## 6. Implementation Order

| Step | File | Action | Dependency |
|------|------|--------|------------|
| 1 | - | `nptdms` 설치 확인 | - |
| 2 | `data/tdms_loader.py` | 신규 작성 | Step 1 |
| 3 | `inference/picker.py` | `pick_tdms()`, `get_probabilities_tdms()` 추가 | Step 2 |
| 4 | `scripts/predict.py` | `--channel` 인자 + TDMS 분기 + `plot_results_das()` | Step 3 |
| 5 | `setup.py` | `extras_require["das"]` 추가 | - |
| 6 | - | 실제 TDMS 파일로 추론 + 시각화 테스트 | Step 4 |

---

## 7. Test Strategy

### 7.1 수동 테스트 (탐색적 실험 특성상)

```bash
# 기본 추론 (채널 0)
python -m scripts.predict --input A2_UTC+0900_DST0_20240618_202544.547.tdms \
    --model checkpoints/best_model.pt

# 특정 채널 + 시각화
python -m scripts.predict --input A2_UTC+0900_DST0_20240618_202544.547.tdms \
    --model checkpoints/best_model.pt --channel 500 --plot

# 다른 채널 비교
python -m scripts.predict --input A2_UTC+0900_DST0_20240618_202544.547.tdms \
    --model checkpoints/best_model.pt --channel 100 --plot
```

### 7.2 검증 포인트

- [ ] TDMS 파일 정상 로드 (1216 채널, 30000 samples)
- [ ] 다운샘플링 결과: 30000 → 3000 samples
- [ ] 3채널 복제 후 shape: (3, 3000)
- [ ] 전처리 후 값 범위 정상 (NaN/Inf 없음)
- [ ] 모델 추론 정상 실행 (no error)
- [ ] 확률 곡선 합 ≈ 1.0 (softmax 출력)
- [ ] 시각화 정상 출력 (4 패널)

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DAS strain rate ≠ velocity | 모델이 의미 있는 P/S 검출 실패 | 탐색적 실험; 확률 곡선 확인 후 판단 |
| 단일성분 복제 한계 | 3성분 편파 정보 부재 | 알려진 한계, 결과 해석에 주의 |
| 1000Hz에서 100Hz로 10x 다운샘플링 | 고주파 정보 손실 | `decimate(zero_phase=True)` 사용 |
| nptdms import 실패 | DAS 기능 사용 불가 | lazy import + 설치 안내 메시지 |

---

## 9. Out of Scope (v1)

- `predict.py --input-dir` TDMS 디렉토리 일괄 처리
- `--channels 100-200` 다채널 범위 지정
- DAS 특화 전처리 (strain rate 보정, spatial filtering, FK filtering)
- DAS 데이터로 모델 fine-tuning
- TDMS 메타데이터 기반 절대 시각 변환
