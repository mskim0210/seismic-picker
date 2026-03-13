import numpy as np


class Compose:
    """여러 증강 변환을 순차적으로 적용."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, waveform, labels):
        for t in self.transforms:
            waveform, labels = t(waveform, labels)
        return waveform, labels


class AddGaussianNoise:
    """Gaussian 노이즈 추가.

    Args:
        snr_range: (min_snr_db, max_snr_db) - 신호 대 잡음비 범위
        probability: 적용 확률
    """

    def __init__(self, snr_range=(5, 30), probability=0.5):
        self.snr_range = snr_range
        self.probability = probability

    def __call__(self, waveform, labels):
        if np.random.random() > self.probability:
            return waveform, labels

        snr_db = np.random.uniform(*self.snr_range)
        signal_power = np.mean(waveform ** 2)

        if signal_power < 1e-10:
            return waveform, labels

        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(*waveform.shape) * np.sqrt(noise_power)
        waveform = waveform + noise.astype(waveform.dtype)
        return waveform, labels


class AmplitudeScale:
    """진폭 랜덤 스케일링.

    Args:
        scale_range: (min_scale, max_scale)
        probability: 적용 확률
    """

    def __init__(self, scale_range=(0.5, 2.0), probability=0.5):
        self.scale_range = scale_range
        self.probability = probability

    def __call__(self, waveform, labels):
        if np.random.random() > self.probability:
            return waveform, labels

        scale = np.random.uniform(*self.scale_range)
        waveform = waveform * scale
        return waveform, labels


class RandomTimeShift:
    """랜덤 시간 이동 (라벨도 함께).

    Args:
        max_shift: 최대 이동 샘플 수
        probability: 적용 확률
    """

    def __init__(self, max_shift=200, probability=0.3):
        self.max_shift = max_shift
        self.probability = probability

    def __call__(self, waveform, labels):
        if np.random.random() > self.probability:
            return waveform, labels

        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        waveform = np.roll(waveform, shift, axis=-1)
        labels = np.roll(labels, shift, axis=-1)

        # 이동된 경계 부분을 0으로 채움
        if shift > 0:
            waveform[:, :shift] = 0
            labels[:, :shift] = 0
            labels[0, :shift] = 1.0  # Noise
        elif shift < 0:
            waveform[:, shift:] = 0
            labels[:, shift:] = 0
            labels[0, shift:] = 1.0  # Noise

        return waveform, labels


class RandomChannelDrop:
    """랜덤 채널 드롭 (결측 성분 시뮬레이션).

    Args:
        probability: 적용 확률
    """

    def __init__(self, probability=0.1):
        self.probability = probability

    def __call__(self, waveform, labels):
        if np.random.random() > self.probability:
            return waveform, labels

        ch = np.random.randint(0, waveform.shape[0])
        waveform[ch] = 0.0
        return waveform, labels


class RandomPolarityFlip:
    """랜덤 극성 반전.

    Args:
        probability: 적용 확률
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, waveform, labels):
        if np.random.random() > self.probability:
            return waveform, labels

        waveform = -waveform
        return waveform, labels


class RandomGapInsertion:
    """랜덤 데이터 결측 구간 삽입.

    실제 계측기 데이터에서 발생하는 데이터 드롭아웃/결측을 시뮬레이션.
    NORSAR TPhasenet 논문의 gap insertion augmentation 참고.

    Args:
        gap_length_range: (min, max) 갭 길이 (samples)
        max_gaps: 한 trace 내 최대 갭 수
        probability: 적용 확률
    """

    def __init__(self, gap_length_range=(50, 500), max_gaps=3, probability=0.2):
        self.gap_length_range = gap_length_range
        self.max_gaps = max_gaps
        self.probability = probability

    def __call__(self, waveform, labels):
        if np.random.random() > self.probability:
            return waveform, labels

        n_samples = waveform.shape[-1]
        n_gaps = np.random.randint(1, self.max_gaps + 1)

        for _ in range(n_gaps):
            gap_len = np.random.randint(*self.gap_length_range)
            max_start = n_samples - gap_len
            if max_start <= 0:
                continue
            start = np.random.randint(0, max_start)
            end = start + gap_len

            waveform[:, start:end] = 0.0
            labels[:, start:end] = 0.0
            labels[0, start:end] = 1.0  # Noise

        return waveform, labels


def get_default_augmentation():
    """기본 데이터 증강 파이프라인 반환."""
    return Compose([
        AddGaussianNoise(snr_range=(5, 30), probability=0.5),
        AmplitudeScale(scale_range=(0.5, 2.0), probability=0.5),
        RandomTimeShift(max_shift=200, probability=0.3),
        RandomChannelDrop(probability=0.1),
        RandomPolarityFlip(probability=0.5),
        RandomGapInsertion(gap_length_range=(50, 500), max_gaps=3, probability=0.2),
    ])
