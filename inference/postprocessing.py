import numpy as np
from scipy.signal import find_peaks


def extract_picks(prob_curves, sampling_rate=100.0, min_height=0.3,
                  min_distance=100, min_prominence=0.1):
    """확률 곡선에서 P파/S파 pick을 추출.

    Args:
        prob_curves: (3, T) numpy array - [Noise, P, S] 확률
        sampling_rate: 샘플링 레이트 (Hz)
        min_height: 최소 peak 높이 (확률 임계값)
        min_distance: peak 간 최소 거리 (samples)
        min_prominence: 최소 prominence (peak 두드러짐 정도)

    Returns:
        picks: list of dict
            {
                'phase': 'P' or 'S',
                'sample_index': int,
                'time_offset_sec': float,
                'confidence': float,
                'uncertainty_sec': float
            }
    """
    picks = []

    for phase_idx, phase_name in [(1, "P"), (2, "S")]:
        curve = prob_curves[phase_idx]

        peaks, properties = find_peaks(
            curve,
            height=min_height,
            distance=min_distance,
            prominence=min_prominence,
            width=True,
        )

        for i, peak_idx in enumerate(peaks):
            confidence = float(properties["peak_heights"][i])

            # uncertainty: peak 너비의 절반 (초 단위)
            width_samples = properties["widths"][i] if "widths" in properties else 0
            uncertainty = float(width_samples / sampling_rate / 2.0)

            picks.append({
                "phase": phase_name,
                "sample_index": int(peak_idx),
                "time_offset_sec": float(peak_idx / sampling_rate),
                "confidence": confidence,
                "uncertainty_sec": uncertainty,
            })

    # 시간순 정렬
    picks.sort(key=lambda p: p["sample_index"])

    # 물리적 제약 적용: S파는 P파 이후에 도착
    picks = _apply_physical_constraints(picks)

    return picks


def _apply_physical_constraints(picks):
    """물리적 제약 적용.

    - S파는 반드시 P파 이후에 도착해야 함
    - 같은 이벤트의 P-S 쌍에서 S가 P보다 앞서면 낮은 신뢰도의 pick 제거
    """
    p_picks = [p for p in picks if p["phase"] == "P"]
    s_picks = [p for p in picks if p["phase"] == "S"]

    if not p_picks or not s_picks:
        return picks

    valid_picks = []

    # P pick마다 이후에 오는 가장 가까운 S pick을 매칭
    used_s = set()
    for p in p_picks:
        valid_picks.append(p)
        for j, s in enumerate(s_picks):
            if j not in used_s and s["sample_index"] > p["sample_index"]:
                valid_picks.append(s)
                used_s.add(j)
                break

    # 매칭되지 않은 S pick 중 P 이후에 위치한 것들 추가
    for j, s in enumerate(s_picks):
        if j not in used_s:
            # P가 없어도 S pick이 단독으로 존재할 수 있음
            if not p_picks or s["sample_index"] > p_picks[0]["sample_index"]:
                valid_picks.append(s)

    valid_picks.sort(key=lambda p: p["sample_index"])
    return valid_picks


def merge_sliding_window_probs(prob_list, window_size, step, total_length):
    """슬라이딩 윈도우 결과를 병합.

    오버랩 구간은 확률을 평균화하여 매끄러운 결과 생성.

    Args:
        prob_list: list of (3, window_size) numpy arrays
        window_size: 윈도우 크기 (samples)
        step: 윈도우 이동 간격 (samples)
        total_length: 원본 파형 전체 길이 (samples)

    Returns:
        merged: (3, total_length) numpy array
    """
    merged = np.zeros((3, total_length), dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)

    for i, probs in enumerate(prob_list):
        start = i * step
        end = min(start + probs.shape[1], total_length)
        length = end - start

        merged[:, start:end] += probs[:, :length]
        counts[start:end] += 1.0

    # 오버랩 구간 평균화
    counts = np.maximum(counts, 1.0)
    merged /= counts[np.newaxis, :]

    return merged.astype(np.float32)
