import numpy as np
from obspy import read as obspy_read
from .preprocessing import preprocess


# 채널 코드 → 성분 인덱스 매핑 (Z=0, N=1, E=2)
CHANNEL_MAP = {
    "Z": 0, "N": 1, "E": 2,
    "1": 1, "2": 2, "3": 0,
}


def _get_component_index(channel_code):
    """채널 코드에서 성분 인덱스 추출.

    BHZ -> Z -> 0, HHN -> N -> 1, EHE -> E -> 2
    """
    last_char = channel_code[-1].upper()
    return CHANNEL_MAP.get(last_char, -1)


def _load_and_align(file_path, target_sampling_rate=100.0):
    """mseed 파일에서 3성분 파형을 로드하고 정렬.

    Args:
        file_path: mseed 파일 경로
        target_sampling_rate: 목표 샘플링 레이트 (Hz)

    Returns:
        aligned: (3, N) numpy array (float64)
        metadata: dict (start_time, station, network, channels)
    """
    st = obspy_read(str(file_path))

    # 채널별로 정렬하여 Z, N, E 순서로 배치
    components = [None, None, None]
    channels_found = []

    for tr in st:
        idx = _get_component_index(tr.stats.channel)
        if idx >= 0:
            components[idx] = tr
            channels_found.append(tr.stats.channel)

    # 누락된 성분 확인
    missing = [i for i, c in enumerate(components) if c is None]
    if len(missing) == 3:
        raise ValueError(f"mseed 파일에서 유효한 3성분을 찾을 수 없습니다: {file_path}")

    # 메타데이터 (첫 번째 유효한 trace에서 추출)
    ref_trace = next(c for c in components if c is not None)
    metadata = {
        "start_time": str(ref_trace.stats.starttime),
        "station": ref_trace.stats.station,
        "network": ref_trace.stats.network,
        "location": ref_trace.stats.location,
        "channels": channels_found,
        "original_sampling_rate": ref_trace.stats.sampling_rate,
    }

    # 각 성분 전처리 (리샘플링 + 누락 채널 0 채움)
    data_arrays = []
    for i, tr in enumerate(components):
        if tr is None:
            n_samples = int(ref_trace.stats.npts * (target_sampling_rate
                                                     / ref_trace.stats.sampling_rate))
            data_arrays.append(np.zeros(n_samples, dtype=np.float64))
        else:
            if abs(tr.stats.sampling_rate - target_sampling_rate) > 0.01:
                tr.resample(target_sampling_rate)
            data_arrays.append(tr.data.astype(np.float64))

    # 모든 성분의 길이를 동일하게 맞춤 (가장 긴 것 기준)
    max_len = max(len(d) for d in data_arrays)
    aligned = np.zeros((3, max_len), dtype=np.float64)
    for i, d in enumerate(data_arrays):
        aligned[i, :len(d)] = d

    return aligned, metadata


def load_mseed(file_path, target_sampling_rate=100.0, target_length=6000,
               config=None):
    """mseed 파일을 로드하여 전처리된 3성분 파형 반환.

    Args:
        file_path: mseed 파일 경로
        target_sampling_rate: 목표 샘플링 레이트 (Hz)
        target_length: 목표 샘플 수 (6000 = 60초 × 100Hz)
        config: data 설정 dict

    Returns:
        waveform: (3, target_length) numpy array (float32)
        metadata: dict (start_time, station, network, channels)
    """
    aligned, metadata = _load_and_align(file_path, target_sampling_rate)

    # 전처리 (demean, detrend, bandpass, normalize)
    aligned = preprocess(aligned, target_sampling_rate, config)

    # target_length에 맞춤 (pad 또는 trim)
    n = aligned.shape[1]
    if n < target_length:
        padded = np.zeros((3, target_length), dtype=np.float32)
        padded[:, :n] = aligned
        waveform = padded
    elif n > target_length:
        waveform = aligned[:, :target_length]
    else:
        waveform = aligned

    return waveform, metadata


def load_mseed_stream(file_path, target_sampling_rate=100.0, config=None):
    """mseed 파일을 가변 길이로 로드 (슬라이딩 윈도우용).

    target_length로 자르지 않고 전체 파형을 반환.

    Args:
        file_path: mseed 파일 경로
        target_sampling_rate: 목표 샘플링 레이트 (Hz)
        config: data 설정 dict

    Returns:
        waveform: (3, N) numpy array (float32)
        metadata: dict
    """
    aligned, metadata = _load_and_align(file_path, target_sampling_rate)

    # 전처리 (demean, detrend, bandpass, normalize)
    aligned = preprocess(aligned, target_sampling_rate, config)

    return aligned, metadata
