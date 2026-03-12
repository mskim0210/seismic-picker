"""DAS TDMS 파일 로더.

1000Hz 단일성분 DAS 데이터를 TPhaseNet 입력 형태(3ch, 100Hz)로 변환.
"""

import numpy as np
from scipy.signal import decimate

from .preprocessing import preprocess


def load_tdms_channel(tdms_path, channel_index=0, target_sampling_rate=100.0,
                      config=None):
    """TDMS 파일에서 단일 DAS 채널을 로드하여 모델 입력 형태로 변환.

    Args:
        tdms_path: TDMS 파일 경로
        channel_index: DAS 채널 인덱스 (0-based)
        target_sampling_rate: 목표 샘플링 레이트 (Hz)
        config: data 설정 dict (전처리 파라미터)

    Returns:
        waveform: (3, N) numpy array (float32) — 3채널 복제, 다운샘플링 후
        metadata: dict
    """
    try:
        from nptdms import TdmsFile
    except ImportError:
        raise ImportError(
            "nptdms가 설치되지 않았습니다. "
            "pip install nptdms 또는 conda install -c conda-forge nptdms"
        )

    # TDMS 파일 읽기
    tdms_file = TdmsFile.read(str(tdms_path))

    # 그룹 찾기 (일반적으로 "Measurement")
    groups = tdms_file.groups()
    if not groups:
        raise ValueError(f"TDMS 파일에 그룹이 없습니다: {tdms_path}")
    group = groups[0]

    # 채널 선택
    channels = group.channels()
    if channel_index < 0 or channel_index >= len(channels):
        raise IndexError(
            f"채널 인덱스 {channel_index}이 범위를 벗어났습니다. "
            f"유효 범위: 0-{len(channels) - 1}"
        )
    channel = channels[channel_index]

    # 원시 데이터 읽기
    raw_data = channel.data.astype(np.float64)

    # 메타데이터 추출
    props = channel.properties
    original_sr = float(props.get("SamplingFrequency",
                                  group.properties.get("SamplingFrequency", 1000)))

    metadata = {
        "source_file": str(tdms_path),
        "channel_index": channel_index,
        "channel_name": channel.name,
        "original_sampling_rate": original_sr,
        "spatial_resolution": float(props.get("SpatialResolution",
                                              group.properties.get("SpatialResolution", 0))),
        "start_position": float(props.get("StartPosition",
                                          group.properties.get("StartPosition", 0))),
        "n_original_samples": len(raw_data),
        "sampling_rate": target_sampling_rate,
        "data_type": "DAS",
        "n_channels_total": len(channels),
    }

    # 다운샘플링
    factor = original_sr / target_sampling_rate
    if factor > 1:
        int_factor = int(round(factor))
        if abs(factor - int_factor) > 0.01:
            raise ValueError(
                f"정수 비율 다운샘플링만 지원합니다. "
                f"원본: {original_sr}Hz, 목표: {target_sampling_rate}Hz, "
                f"비율: {factor}"
            )
        data = decimate(raw_data, int_factor, zero_phase=True)
    else:
        data = raw_data

    metadata["n_samples"] = len(data)

    # 단일성분 → 3채널 복제
    waveform = np.stack([data, data, data], axis=0)

    # 전처리 (demean, detrend, bandpass, normalize)
    waveform = preprocess(waveform, target_sampling_rate, config)

    return waveform, metadata
