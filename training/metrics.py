import numpy as np
from scipy.signal import find_peaks


def compute_pick_metrics(pred_probs, true_labels, threshold=0.3,
                         tolerance_samples=50, sampling_rate=100.0):
    """배치 단위 pick 정확도 계산.

    Args:
        pred_probs:  (B, 3, T) numpy array - 모델 출력
        true_labels: (B, 3, T) numpy array - Gaussian 라벨
        threshold: 최소 peak 높이
        tolerance_samples: 허용 오차 (50 samples = 0.5초)
        sampling_rate: 샘플링 레이트

    Returns:
        dict: P/S별 precision, recall, F1, 평균 잔차
    """
    results = {"P": _init_counts(), "S": _init_counts()}

    batch_size = pred_probs.shape[0]

    for b in range(batch_size):
        for phase_idx, phase_name in [(1, "P"), (2, "S")]:
            pred_curve = pred_probs[b, phase_idx]
            true_curve = true_labels[b, phase_idx]

            # 예측 pick
            pred_peaks, _ = find_peaks(pred_curve, height=threshold,
                                       distance=50)

            # 실제 pick (Gaussian 피크)
            true_peak = np.argmax(true_curve)
            has_true = true_curve[true_peak] > 0.5 if true_peak > 0 else False

            if len(pred_peaks) > 0 and has_true:
                # 가장 가까운 예측 pick과 비교
                distances = np.abs(pred_peaks - true_peak)
                closest_idx = np.argmin(distances)
                min_dist = distances[closest_idx]

                if min_dist <= tolerance_samples:
                    results[phase_name]["tp"] += 1
                    residual = (pred_peaks[closest_idx] - true_peak) / sampling_rate
                    results[phase_name]["residuals"].append(residual)
                    # 나머지 예측은 FP
                    results[phase_name]["fp"] += len(pred_peaks) - 1
                else:
                    results[phase_name]["fp"] += len(pred_peaks)
                    results[phase_name]["fn"] += 1
            elif len(pred_peaks) > 0 and not has_true:
                results[phase_name]["fp"] += len(pred_peaks)
            elif len(pred_peaks) == 0 and has_true:
                results[phase_name]["fn"] += 1

    return _compute_scores(results)


def _init_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "residuals": []}


def _compute_scores(results):
    """TP/FP/FN에서 precision, recall, F1 계산."""
    scores = {}
    for phase in ["P", "S"]:
        r = results[phase]
        tp, fp, fn = r["tp"], r["fp"], r["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        residuals = r["residuals"]
        mean_res = np.mean(residuals) if residuals else 0.0
        std_res = np.std(residuals) if residuals else 0.0

        prefix = phase.lower()
        scores[f"{prefix}_precision"] = precision
        scores[f"{prefix}_recall"] = recall
        scores[f"{prefix}_f1"] = f1
        scores[f"{prefix}_mean_residual_sec"] = mean_res
        scores[f"{prefix}_std_residual_sec"] = std_res
        scores[f"{prefix}_tp"] = tp
        scores[f"{prefix}_fp"] = fp
        scores[f"{prefix}_fn"] = fn

    return scores
