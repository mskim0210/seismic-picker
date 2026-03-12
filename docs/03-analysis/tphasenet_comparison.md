# TPhasenet (NORSAR) vs seismic-picker 모델 비교 분석

## 논문 정보

- **제목**: Deep learning models for regional phase detection on seismic stations in Northern Europe and the European Arctic
- **저자**: Erik Myklebust, Andreas Köhler (NORSAR)
- **출판**: Geophysical Journal International, Vol. 239, Issue 2, 2024
- **DOI**: https://doi.org/10.1093/gji/ggae306
- **GitHub**: https://github.com/NORSAR-official/tphasenet

## 핵심 차이점 비교표

| 항목 | NORSAR TPhasenet | seismic-picker |
|---|---|---|
| **입력 길이** | 12,000 samples (300초 @ 40Hz) | 6,000 samples (60초 @ 100Hz) |
| **샘플링 레이트** | 40 Hz | 100 Hz |
| **Encoder 깊이** | 7 levels | 7 levels + bottleneck |
| **필터 수** | [64, 64, 128, 128, 256, 256, 512] | [8, 16, 32, 64, 128, 256, 512] |
| **Convolution 블록** | ResNet Block (residual + projection shortcut) | Plain Conv1d + BN + ReLU |
| **Pooling stride** | 2 (MaxPool, size 4) | 4 (strided convolution) |
| **Activation** | Swish | ReLU (conv), GELU (transformer FF) |
| **Transformer 위치** | Skip connection (decoder 단계) | Encoder 깊은 레벨 (level 3+) |
| **Transformer 방식** | BiLSTM → 1x1 Conv → Cross-attention | Self-attention + positional encoding |
| **Attention type** | "across" 또는 "downstep" | Encoder 내부 self-attention |
| **RNN 사용** | BiLSTM (skip connection 처리) | 없음 |
| **Dropout** | 0.4 | 0.1 (transformer only) |
| **학습 데이터** | NORSAR/Helsinki/CTBTO (~151K traces, 북유럽) | STEAD (~1.27M traces, 글로벌) |
| **Phase 유형** | 9종 regional → 3 class 축소 | 3 class (Noise, P, S) |
| **Loss** | CCE, weights [0.05, 0.40, 0.55] | Weighted CCE, weights [1.0, 30.0, 30.0] |
| **Label smoothing** | Gaussian (ramp=11) | Gaussian (sigma=20 samples) |
| **파라미터 수** | 수백만 (필터 64~512) | ~517K (필터 8~512) |
| **Sliding window 병합** | Median stacking (step 10s) | Weighted averaging (50% overlap) |

## 상세 아키텍처 비교

### 1. Transformer 적용 방식

이 두 모델의 가장 핵심적인 아키텍처 차이이다.

**NORSAR TPhasenet:**
- Skip connection에서 BiLSTM + Transformer cross-attention 적용
- `_att_block` 메서드에서 encoder skip feature를 BiLSTM으로 처리한 후, 1x1 Conv 투영을 거쳐 Transformer block으로 전달
- Decoder가 encoder feature를 **선택적으로 attend**할 수 있음
- 두 가지 모드 지원:
  - `"across"` (기본값): Decoding 시 skip feature에 self-attention 적용 후 decoder와 concatenation
  - `"downstep"`: Encoding 시 이전 레벨 feature에 attention 적용

**seismic-picker:**
- Encoder 깊은 레벨(level 3 이상)에서 self-attention 적용
- Sinusoidal positional encoding 사용
- Multi-head self-attention (4~8 heads)
- Skip connection은 단순 concatenation 방식 유지

**분석**: NORSAR 방식이 encoder-decoder 간 정보 전달을 학습 가능하게 만들어 더 정교하다. 반면 seismic-picker는 encoder 내부에서만 global dependency를 학습한다.

### 2. Backbone 구조

**NORSAR TPhasenet:**
- ResNet Block: pre-activation residual block + projection shortcut (1x1 conv)
- Swish activation (smooth, non-monotonic)
- 시작 필터 64채널로 높은 모델 용량

**seismic-picker:**
- Plain Conv1d + BatchNorm + ReLU
- 시작 필터 8채널로 경량화 설계
- 총 파라미터 ~517K

**분석**: NORSAR 모델이 훨씬 큰 모델 용량을 가지며, residual connection으로 깊은 네트워크에서도 gradient flow가 안정적이다. seismic-picker는 경량화에 초점을 맞추어 빠른 추론이 가능하다.

### 3. 입력 윈도우 및 샘플링 전략

**NORSAR TPhasenet:**
- 300초 윈도우 @ 40Hz = 12,000 samples
- 원본 데이터 540초에서 300초를 random crop하여 학습
- Regional phase는 원거리에서 넓은 시간 범위에 도달하므로 긴 윈도우 필요

**seismic-picker:**
- 60초 윈도우 @ 100Hz = 6,000 samples
- STEAD 데이터 기본 포맷 그대로 사용
- Local event에 적합, 더 높은 시간 해상도

**분석**: 목적이 다르다. NORSAR는 regional/near-teleseismic (최대 5,000km) 이벤트를 대상으로 하여 긴 윈도우가 필수적이다. seismic-picker는 local event 위주로 높은 시간 해상도를 우선한다.

### 4. 데이터 및 학습 전략

**NORSAR TPhasenet:**
- 데이터: NORSAR, Helsinki, CTBTO 카탈로그 (~151K traces, ~100K events)
- 지역: 북유럽/북극 (ARCES, FINES, SPITS, NORES 관측소)
- Regional phase 9종 (Pn, Pg, P, Sn, Sg, S, Pb, Sb, D)을 3 class로 축소
- 연도 기반 split (2000-2020 / 2021 / 2022)
- Loss weights: noise 크게 줄임 [0.05, 0.40, 0.55]

**seismic-picker:**
- 데이터: STEAD (~1.27M traces, 글로벌)
- 3 class (Noise, P, S) 직접 학습
- Hash 기반 deterministic split (80/10/10)
- Loss weights: phase에 높은 가중치 [1.0, 30.0, 30.0]

**분석**: NORSAR는 지역 특화 모델로 regional phase 구분이 핵심 기여이다. seismic-picker는 범용 글로벌 모델로 데이터 규모가 8배 이상 크다.

### 5. 데이터 증강(Augmentation)

| 기법 | NORSAR TPhasenet | seismic-picker |
|---|---|---|
| Gaussian noise | 30% | 50% (SNR 5-30dB) |
| Event superposition | 30% (핵심 기법) | 없음 |
| Channel drop | 20% | 10% |
| Gap insertion | 20% | 없음 |
| Amplitude scale | 없음 | 50% (0.5-2.0x) |
| Time shift | 없음 (random crop으로 대체) | 30% (±200 samples) |
| Polarity flip | 없음 | 50% |
| Taper | Tukey (alpha=0.01) | 없음 |

**분석**: NORSAR의 **Event superposition**은 두 이벤트를 겹쳐 학습하여 밀집 이벤트 구간에서 성능을 향상시키는 독창적 기법이다. seismic-picker는 polarity flip과 amplitude scaling으로 일반화 성능을 높이는 전략을 사용한다.

### 6. 추론 파이프라인

**NORSAR TPhasenet:**
- Sliding window (300초), step 10초
- 겹치는 구간은 **median stacking**으로 병합 (mean, std, 25th percentile도 지원)
- Peak detection threshold: P=0.6, S=0.5

**seismic-picker:**
- Sliding window (60초), step 50% overlap (30초)
- 겹치는 구간은 **weighted averaging**으로 병합
- Peak detection: min_height=0.3, min_distance=100 samples, min_prominence=0.1

**분석**: Median stacking은 outlier에 강건하여 연속 모니터링에 유리하다. Weighted averaging은 smoother한 결과를 제공한다.

## 성능 비교

### NORSAR TPhasenet (북유럽 regional data)

| Phase | Precision | Recall | F1 |
|---|---|---|---|
| P | - | 0.88 | - |
| S | - | 0.86 | - |

- 기존 ARCES array detector(FKX) 대비 detection rate 증가, false detection 감소

### seismic-picker (STEAD global data)

| Phase | Precision | Recall | F1 | Residual (sec) |
|---|---|---|---|---|
| P | 0.9708 | 0.9951 | 0.9828 | -0.007 ± 0.039 |
| S | 0.9569 | 0.9738 | 0.9653 | +0.004 ± 0.111 |

**주의**: 데이터셋과 평가 기준이 다르므로 직접적인 수치 비교는 의미가 제한적이다. NORSAR는 regional phase라는 더 어려운 문제를 다루고 있다.

## seismic-picker에 적용 가능한 개선 사항

### 높은 우선순위

1. **Skip connection에 attention 도입**
   - 단순 concatenation 대신 cross-attention 또는 BiLSTM 추가
   - Decoder가 encoder feature를 선택적으로 활용할 수 있게 됨
   - NORSAR 논문에서 가장 큰 성능 향상 요인

2. **ResNet block 적용**
   - Plain conv → residual block으로 교체
   - Gradient flow 개선으로 깊은 네트워크 학습 안정화
   - Projection shortcut으로 채널 변환 시에도 residual path 유지

### 중간 우선순위

3. **Event superposition augmentation**
   - 두 이벤트 파형을 겹쳐서 학습 데이터 생성
   - Label은 두 이벤트의 max 취함
   - 밀집 이벤트 구간 성능 향상에 효과적

4. **Median stacking 추론**
   - 연속 데이터 처리 시 outlier에 강건한 병합 방식
   - 특히 noise가 많은 환경에서 유리

### 낮은 우선순위

5. **Swish activation 실험**
   - ReLU → Swish로 교체 시 smooth한 gradient로 학습 안정성 향상 가능
   - 모델 크기 변경 없이 적용 가능

6. **Gap insertion augmentation**
   - 실제 데이터에서 발생하는 데이터 결측 구간에 대한 robustness 향상

## 결론

두 모델은 같은 TPhasenet이라는 이름을 공유하지만, Transformer를 U-Net에 통합하는 방식이 근본적으로 다르다.

- **NORSAR**: Transformer를 **skip connection 경로**에 배치하여 encoder-decoder 간 정보 전달을 학습 가능하게 함 (BiLSTM + cross-attention)
- **seismic-picker**: Transformer를 **encoder 내부**에 배치하여 long-range temporal dependency를 직접 학습

NORSAR 모델은 큰 모델 용량과 긴 윈도우로 regional phase 검출에 특화되었고, seismic-picker는 경량 모델로 글로벌 데이터 기반의 범용 P/S 검출에 초점을 맞추고 있다. Skip connection attention과 ResNet block의 도입이 현재 프로젝트의 성능 향상에 가장 유망한 개선 방향이다.
