import numpy as np


def generate_gaussian_label(n_samples, arrival_sample, sigma=20):
    """단일 위상의 Gaussian 라벨 생성.

    Args:
        n_samples: 전체 샘플 수
        arrival_sample: 도착 시각의 샘플 인덱스 (None이면 0 배열)
        sigma: Gaussian 너비 (samples, 20=0.2초 at 100Hz)

    Returns:
        (n_samples,) numpy array
    """
    if arrival_sample is None or np.isnan(arrival_sample):
        return np.zeros(n_samples, dtype=np.float32)

    t = np.arange(n_samples, dtype=np.float32)
    label = np.exp(-((t - arrival_sample) ** 2) / (2.0 * sigma ** 2))
    return label


def generate_labels(n_samples, p_sample, s_sample, sigma=20):
    """3-class 라벨 배열 생성: [Noise, P, S].

    Args:
        n_samples: 전체 샘플 수
        p_sample: P파 도착 샘플 인덱스 (None 가능)
        s_sample: S파 도착 샘플 인덱스 (None 가능)
        sigma: Gaussian 너비 (samples)

    Returns:
        (3, n_samples) numpy array - [Noise, P, S] 확률
    """
    p_prob = generate_gaussian_label(n_samples, p_sample, sigma)
    s_prob = generate_gaussian_label(n_samples, s_sample, sigma)

    # Noise = 1.0 - P - S, [0, 1]로 클램핑
    noise_prob = np.clip(1.0 - p_prob - s_prob, 0.0, 1.0)

    labels = np.stack([noise_prob, p_prob, s_prob], axis=0)
    return labels.astype(np.float32)
