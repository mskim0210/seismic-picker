# Plan: NORSAR 기반 모델 개선

## Executive Summary

| 관점 | 내용 |
|------|------|
| **Problem** | 현재 TPhaseNet은 plain convolution + 단순 skip concatenation 구조로, NORSAR TPhasenet 대비 모델 표현력과 gradient flow에서 한계가 있음 |
| **Solution** | NORSAR 논문의 핵심 기법 4가지(Swish, ResNet block, Skip attention, Gap augmentation)를 incremental 방식으로 통합 |
| **기능/UX 효과** | 모델 아키텍처 고도화로 P/S 검출 정밀도 향상, 실환경 데이터 결측에 대한 robustness 확보 |
| **핵심 가치** | KIGAM 연구용 모델의 최신 기법 적용으로 논문/발표 수준의 모델 품질 달성 |

---

## 1. User Intent Discovery

### 핵심 문제
NORSAR TPhasenet 논문과의 비교 분석(`docs/03-analysis/tphasenet_comparison.md`)에서 도출된 6가지 개선사항 중 4가지를 현재 seismic-picker 모델에 적용하여 아키텍처를 고도화한다.

### 대상 사용자
- KIGAM 내부 지진 연구원
- GPU 자원 제약을 고려한 경량 모델 유지 (현재 ~517K params)

### 성공 기준
- 4가지 개선사항 모두 구현 완료
- 기존 pytest 82개 전수 통과
- 기존 모델과 하위호환성 유지 (config 기본값으로 기존 동작 보장)

---

## 2. Scope

### In Scope

| # | 개선사항 | 우선순위 | 영향 파일 |
|---|---------|---------|----------|
| 1 | Swish(SiLU) activation | LOW | `conv_blocks.py`, `encoder.py`, config |
| 2 | ResNet block (residual + projection shortcut) | HIGH | `conv_blocks.py` |
| 3 | Skip connection attention (BiLSTM + cross-attention) | HIGH | 새 `skip_attention.py`, `decoder.py`, `tphasenet.py`, config |
| 4 | Gap insertion augmentation | LOW | `augmentation.py` |

### Out of Scope

| # | 개선사항 | 사유 |
|---|---------|------|
| 5 | Event superposition augmentation | 첫 버전에서 제외 (별도 STEAD 데이터 로딩 필요) |
| 6 | Median stacking 추론 | 첫 버전에서 제외 (추론 파이프라인 변경 범위 큼) |

---

## 3. Alternatives Explored

### Approach A: Incremental Integration (선택됨)
- 우선순위순으로 하나씩 구현, 단계별 테스트 통과 확인
- **순서**: Swish → ResNet block → Skip attention → Gap insertion
- **장점**: 각 변경의 영향을 독립적으로 파악 가능, 문제 발생 시 원인 추적 용이
- **단점**: 시간이 더 걸림

### Approach B: Layer-by-Layer Batch
- 관련 기능끼리 묶어서 배치 구현 (모델구조 일괄 → augmentation 일괄)
- 기각 사유: 개별 기여도 파악이 어려움

### Approach C: Config-Driven Toggle
- 모든 개선사항을 config 플래그로 on/off
- 기각 사유: 코드 복잡도 증가, KIGAM 연구용이므로 과도한 유연성 불필요

---

## 4. Implementation Steps

### Step 1: Swish Activation
- `conv_blocks.py`: `DownBlock`, `UpBlock`에서 `nn.ReLU` → `nn.SiLU` 교체
- `encoder.py`: `input_conv`, `bottleneck_conv`의 `nn.ReLU` → `nn.SiLU` 교체
- `config/default.yaml`, `config/defaults.py`: `model.activation: "silu"` 추가
- `tphasenet.py`: `from_config()`에 activation 파라미터 전달
- 테스트 통과 확인

### Step 2: ResNet Block
- `conv_blocks.py` `DownBlock`:
  - 입력에서 출력으로의 residual path 추가
  - 채널 수 또는 시간 길이가 다른 경우 projection shortcut (1x1 Conv1d + stride)
  - `skip = conv1 출력` (기존과 동일), `out = conv2 출력 + shortcut(입력)`
- `conv_blocks.py` `UpBlock`:
  - concat+conv 후 residual connection 추가
  - projection shortcut으로 채널 수 맞춤
- 테스트 통과 확인

### Step 3: Skip Connection Attention
- `models/skip_attention.py` 새 파일 생성:
  ```
  SkipAttentionBlock:
    - BiLSTM(input_dim, hidden_dim) → skip feature의 temporal context 학습
    - 1x1 Conv1d projection → attention 차원 맞춤
    - nn.MultiheadAttention(cross-attention)
      - query: skip feature (encoder)
      - key/value: decoder feature (upsampled)
    - residual connection
  ```
- `decoder.py`:
  - `SkipAttentionBlock` 리스트를 `nn.ModuleList`로 관리
  - `forward()`에서 skip을 UpBlock에 넘기기 전에 attention 적용
  - decoder feature(x)를 attention의 key/value로 사용
- `tphasenet.py`: `from_config()`에 skip attention 파라미터 전달
- `config`: `model.skip_attention: true`, `model.lstm_hidden: 64` 추가
- 테스트 통과 확인

### Step 4: Gap Insertion Augmentation
- `augmentation.py`에 `RandomGapInsertion` 클래스 추가:
  - `gap_length_range: (50, 500)` samples
  - `max_gaps: 3` (한 trace 내 최대 gap 수)
  - `probability: 0.2`
  - gap 구간의 waveform을 0으로, label을 Noise=1.0으로 설정
- `get_default_augmentation()`에 추가
- 테스트 통과 확인

---

## 5. Config Changes

### `config/default.yaml` 추가 항목
```yaml
model:
  activation: "silu"        # "silu" (Swish) or "relu"
  skip_attention: true       # Skip connection attention 사용 여부
  lstm_hidden: 64            # SkipAttentionBlock BiLSTM hidden size
```

### `config/defaults.py` 동기화
```python
"model": {
    ...
    "activation": "silu",
    "skip_attention": True,
    "lstm_hidden": 64,
}
```

---

## 6. Data Flow

```
입력 (B, 3, 6000)  ──[Gap Insertion 적용 가능]──
    │
    ▼
Encoder.input_conv [SiLU]  →  (B, 8, 6000)
    │
    ├── DownBlock level 0 [ResNet + SiLU]
    │   ├── skip0 (B, 8, 6000)
    │   └── out (B, 8, 3000)
    │
    ├── DownBlock level 1 [ResNet + SiLU]
    │   ├── skip1 (B, 16, 3000)
    │   └── out (B, 16, 1500)
    │
    ├── ... (levels 2~3, Transformer at level 3+)
    │
    ├── Bottleneck [SiLU + Transformer]  →  (B, C_max, T_min)
    │
    ▼
Decoder
    ├── x = UpConv(bottleneck)
    ├── skip = SkipAttention(skip_N, query=skip_N, kv=x)
    ├── x = UpBlock(x, attended_skip) [ResNet + SiLU]
    ├── ... (역순 반복)
    ▼
output_conv → softmax → (B, 3, 6000)
```

---

## 7. YAGNI Review

### 포함 항목 (4개)
- [x] Swish activation — 단순 교체, 즉각 효과
- [x] ResNet block — gradient flow 개선, 학습 안정성
- [x] Skip connection attention — NORSAR 핵심 기법, 가장 큰 성능 향상 기대
- [x] Gap insertion augmentation — 실환경 데이터 결측 대응

### 제외 항목 (2개)
- [ ] Event superposition — 별도 데이터 로딩 로직 필요, 차기 버전으로 연기
- [ ] Median stacking — 추론 파이프라인 전면 변경 필요, 차기 버전으로 연기

### YAGNI 원칙 적용
- activation factory 패턴 불필요 → config 문자열로 직접 선택 (`"silu"` / `"relu"`)
- SkipAttention on/off만 지원, attention 방식 선택지(across/downstep) 불필요
- Gap insertion 파라미터는 하드코딩 기본값 사용, config 노출 불필요

---

## 8. Brainstorming Log

| Phase | 결정 | 근거 |
|-------|------|------|
| Phase 1 | 6가지 NORSAR 기법 전반 적용 목표 | 체계적 모델 고도화를 위해 |
| Phase 1 | KIGAM 내부 연구용 | GPU 제약 고려, 경량 모델 유지 |
| Phase 1 | 구현 + 테스트 통과가 성공 기준 | 재학습은 별도 진행 |
| Phase 2 | Incremental 전략 선택 | 변경 영향 추적 용이 |
| Phase 3 | 4개 선택, 2개 제외 | Event superposition/Median stacking은 차기 버전 |
| Phase 4 | Swish → ResNet → Skip Attention → Gap 순서 | 의존성 및 복잡도 순 |

---

## 9. Risks & Mitigations

| 리스크 | 영향 | 완화 방안 |
|--------|------|----------|
| Skip attention으로 파라미터 수 급증 | 학습/추론 속도 저하 | lstm_hidden=64로 제한, 필요시 축소 |
| ResNet block의 projection shortcut 차원 불일치 | 런타임 오류 | 단위 테스트로 shape 검증 |
| 기존 체크포인트 비호환 | 재학습 필요 | config 기본값으로 기존 동작 보장은 유지하되, 새 구조 사용 시 재학습 필수로 문서화 |
| BiLSTM 추가로 순차 처리 병목 | GPU 활용률 저하 | hidden size 제한, 필요시 GRU로 대체 |

---

## 10. Next Steps

1. `/pdca design model-improvements` — 상세 설계 문서 작성
2. 각 Step별 구현 및 pytest 통과 확인
3. 전체 통합 후 파라미터 수 비교 (기존 ~517K vs 개선 후)
4. STEAD 재학습 및 성능 비교 (별도 진행)
