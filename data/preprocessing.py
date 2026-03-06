import numpy as np
from scipy.signal import butter, sosfiltfilt


def demean(data):
    """채널별 평균 제거.

    Args:
        data: (n_channels, n_samples) numpy array
    Returns:
        demeaned data
    """
    return data - np.mean(data, axis=1, keepdims=True)


def detrend(data):
    """채널별 선형 트렌드 제거.

    Args:
        data: (n_channels, n_samples) numpy array
    Returns:
        detrended data
    """
    n_samples = data.shape[1]
    x = np.arange(n_samples, dtype=np.float64)
    result = np.empty_like(data)
    for i in range(data.shape[0]):
        coeffs = np.polyfit(x, data[i], 1)
        trend = np.polyval(coeffs, x)
        result[i] = data[i] - trend
    return result


def bandpass_filter(data, freq_min=0.5, freq_max=45.0, sampling_rate=100.0,
                    corners=4):
    """Butterworth 밴드패스 필터.

    Args:
        data: (n_channels, n_samples) numpy array
        freq_min: 최소 주파수 (Hz)
        freq_max: 최대 주파수 (Hz)
        sampling_rate: 샘플링 레이트 (Hz)
        corners: 필터 차수
    Returns:
        filtered data
    """
    nyquist = sampling_rate / 2.0
    low = freq_min / nyquist
    high = freq_max / nyquist
    sos = butter(corners, [low, high], btype="band", output="sos")
    result = np.empty_like(data)
    for i in range(data.shape[0]):
        result[i] = sosfiltfilt(sos, data[i])
    return result


def normalize(data, method="std", epsilon=1e-8):
    """파형 정규화.

    Args:
        data: (n_channels, n_samples) numpy array
        method: 'std' (표준편차), 'peak' (최대값), 'minmax'
        epsilon: 0 나누기 방지
    Returns:
        normalized data
    """
    if method == "std":
        std = np.std(data) + epsilon
        return data / std
    elif method == "peak":
        peak = np.max(np.abs(data)) + epsilon
        return data / peak
    elif method == "minmax":
        vmin = np.min(data)
        vmax = np.max(data)
        return (data - vmin) / (vmax - vmin + epsilon)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess(data, sampling_rate=100.0, config=None):
    """전체 전처리 파이프라인.

    Args:
        data: (n_channels, n_samples) numpy array
        sampling_rate: 현재 샘플링 레이트
        config: data 설정 dict (없으면 기본값 사용)
    Returns:
        preprocessed data: (n_channels, n_samples) numpy array
    """
    if config is None:
        config = {}

    filter_cfg = config.get("filter", {})
    norm_cfg = config.get("normalize", {})

    data = data.astype(np.float64)

    # 1. Demean
    data = demean(data)

    # 2. Detrend
    data = detrend(data)

    # 3. Bandpass filter
    if filter_cfg.get("enabled", True):
        freq_min = filter_cfg.get("freq_min", 0.5)
        freq_max = filter_cfg.get("freq_max", 45.0)
        corners = filter_cfg.get("corners", 4)
        data = bandpass_filter(data, freq_min, freq_max, sampling_rate, corners)

    # 4. Normalize
    norm_method = norm_cfg.get("method", "std")
    epsilon = norm_cfg.get("epsilon", 1e-8)
    data = normalize(data, norm_method, epsilon)

    return data.astype(np.float32)
