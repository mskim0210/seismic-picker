# DAS Inference Plan

## Executive Summary

| Perspective | Description |
|-------------|-------------|
| **Problem** | 기존 seismic-picker는 3성분 mseed 지진파만 지원하며, 1000Hz 단일성분 DAS(TDMS) 데이터를 처리할 수 없음 |
| **Solution** | TDMS 로더 신규 작성 + 기존 TPhaseNet 파이프라인 재활용으로 DAS 데이터 추론 지원 |
| **Function UX Effect** | 단일 메서드 호출(`pick_tdms()`)로 TDMS → P/S pick 결과 + 시각화 |
| **Core Value** | 추가 학습 없이 기존 모델로 DAS 데이터 적용 가능성 검증 (탐색적 실험) |

---

## 1. User Intent Discovery

### Core Problem
기존 TPhaseNet 모델(STEAD 학습, 100Hz 3성분)을 1000Hz 단일성분 DAS 데이터에 적용하여, 모델이 DAS 환경에서도 P/S파를 검출할 수 있는지 탐색적으로 테스트하고 싶음.

### Target Users
- 연구자 (KIGAM 지진 연구원)
- DAS 데이터를 보유한 지구물리 연구자

### Success Criteria
- TDMS 파일을 읽어 TPhaseNet으로 추론이 정상 실행됨
- P/S 확률 곡선이 시각화로 확인 가능
- 기존 코드 구조를 최소한으로 변경

### Constraints
- 재학습 없이 기존 체크포인트 그대로 사용
- 단일성분 → 3채널 복제 (workaround)
- 1000Hz → 100Hz 다운샘플링 필수

---

## 2. Alternatives Explored

### Approach A: 기존 파이프라인 통합 -- **Selected**
- **Pros**: 코드 재사용 극대화, 최소 변경, 빠른 구현
- **Cons**: DAS 특성(strain rate, 단일성분) 최적화 없음
- **Best for**: 탐색적 실험, 기존 모델 검증

### Approach B: 별도 DAS 파이프라인
- **Pros**: DAS 특화 전처리/후처리 가능
- **Cons**: 코드 중복, 구현 비용 높음
- **Best for**: DAS 전용 시스템 구축 시

### Approach C: SeisBench DAS 모델 래퍼
- **Pros**: 커뮤니티 모델 즉시 사용
- **Cons**: SeisBench에 DAS 전용 모델 부재
- **Best for**: 향후 DAS 모델이 SeisBench에 추가된 경우

---

## 3. YAGNI Review

### Included (v1)
- [x] TDMS 로더 (`data/tdms_loader.py`) — nptdms 읽기, 다운샘플링, 3채널 복제, 전처리
- [x] 시각화 (`--plot`) — DAS 파형 + 확률 곡선 + pick 마커

### Deferred (v2+)
- [ ] `predict.py --input *.tdms` CLI 확장
- [ ] 다채널 일괄처리 (`--channels 100-200`)
- [ ] DAS 특화 전처리 (strain rate 보정, spatial filtering)
- [ ] DAS 전용 모델 fine-tuning

---

## 4. Architecture

### 4.1 Data Flow

```
TDMS 파일 (1000Hz, 1ch, int16, 30000 samples)
    │
    ▼
tdms_loader.py
    ├─ nptdms로 읽기 (채널 인덱스 지정)
    ├─ int16 → float32 정규화
    ├─ 1000Hz → 100Hz 다운샘플링 (scipy.signal.decimate, factor=10)
    ├─ 단일성분 → 3채널 복제 [ch, ch, ch]
    └─ 전처리 (demean, detrend, bandpass 0.5-45Hz, normalize)
          │
          ▼
    numpy array (3, 3000)  ← 30초 × 100Hz
          │
          ▼
기존 SeismicPicker._infer_single()
    ├─ zero-pad → (3, 6000)
    └─ TPhaseNet 추론
          │
          ▼
    확률 곡선 (3, 3000) [Noise, P, S]
          │
    ├─ extract_picks() → P/S pick 리스트
    └─ 시각화 (matplotlib)
```

### 4.2 Component Changes

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `data/tdms_loader.py` | **신규** | TDMS 읽기, 다운샘플링, 3채널 복제, 전처리 |
| `inference/picker.py` | **수정** | `pick_tdms()`, `get_probabilities_tdms()` 메서드 추가 |
| `scripts/predict.py` | **최소 수정** | `.tdms` 확장자 감지 시 `pick_tdms()` 호출 |

### 4.3 tdms_loader.py 주요 함수

```python
def load_tdms_channel(
    tdms_path: str,
    channel_index: int = 0,
    target_sampling_rate: float = 100.0,
    config: dict = None,
) -> Tuple[np.ndarray, dict]:
    """
    TDMS 파일에서 단일 DAS 채널을 로드하여 모델 입력 형태로 변환.

    Returns:
        waveform: (3, N) numpy array (3채널 복제, 다운샘플링 후)
        metadata: dict (sampling_rate, n_samples, channel_index, ...)
    """
```

---

## 5. Dependencies

| 패키지 | 용도 | 설치 |
|--------|------|------|
| `nptdms` | TDMS 파일 읽기 | `pip install nptdms` (또는 `conda install -c conda-forge nptdms`) |
| `scipy` | `decimate()` 다운샘플링 | 이미 설치됨 |

---

## 6. Implementation Order

1. `nptdms` 설치 확인
2. `data/tdms_loader.py` 작성
3. `inference/picker.py`에 `pick_tdms()` 추가
4. `scripts/predict.py`에 TDMS 분기 추가
5. 실제 TDMS 파일로 추론 테스트 + 시각화 확인

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DAS strain rate vs 지진계 velocity 차이 | 모델이 P/S를 검출하지 못할 수 있음 | 탐색적 실험이므로 결과 확인 후 판단 |
| 단일성분 3채널 복제의 한계 | 3성분 정보 부재로 성능 저하 | 알려진 한계, 결과 해석 시 고려 |
| 다운샘플링 aliasing | 고주파 정보 손실 | decimate()의 anti-aliasing 필터 사용 |

---

## 8. Brainstorming Log

| Phase | Decision | Rationale |
|-------|----------|-----------|
| Phase 1 Q1 | 기존 모델 테스트 | 재학습 없이 빠르게 적용 가능성 확인 |
| Phase 1 Q2 | 단일성분 3채널 복제 | 모델 입력 형태(3ch) 맞추기 위한 최소한의 방법 |
| Phase 2 | Approach A (기존 파이프라인 통합) | 코드 재사용, 최소 변경, 빠른 실험 |
| Phase 3 | TDMS 로더 + 시각화만 포함 | CLI 확장/다채널 일괄처리는 불필요 |
