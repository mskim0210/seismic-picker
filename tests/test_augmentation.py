"""Tests for data/augmentation.py"""

import numpy as np
import pytest
from data.augmentation import (
    Compose,
    AddGaussianNoise,
    AmplitudeScale,
    RandomTimeShift,
    RandomChannelDrop,
    RandomPolarityFlip,
    get_default_augmentation,
)
from data.label_utils import generate_labels


@pytest.fixture
def waveform_and_labels():
    np.random.seed(0)
    waveform = np.random.randn(3, 6000).astype(np.float32)
    labels = generate_labels(6000, p_sample=2000, s_sample=4000)
    return waveform, labels


class TestAddGaussianNoise:
    def test_shape_preserved(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = AddGaussianNoise(probability=1.0)
        result_wf, result_lb = aug(wf.copy(), lb.copy())
        assert result_wf.shape == wf.shape
        assert result_lb.shape == lb.shape

    def test_noise_added(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = AddGaussianNoise(probability=1.0, snr_range=(5, 5))
        result_wf, _ = aug(wf.copy(), lb.copy())
        assert not np.allclose(result_wf, wf)

    def test_no_change_when_prob_zero(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = AddGaussianNoise(probability=0.0)
        result_wf, _ = aug(wf.copy(), lb.copy())
        np.testing.assert_array_equal(result_wf, wf)


class TestAmplitudeScale:
    def test_shape_preserved(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = AmplitudeScale(probability=1.0)
        result_wf, result_lb = aug(wf.copy(), lb.copy())
        assert result_wf.shape == wf.shape

    def test_labels_unchanged(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = AmplitudeScale(probability=1.0)
        _, result_lb = aug(wf.copy(), lb.copy())
        np.testing.assert_array_equal(result_lb, lb)


class TestRandomTimeShift:
    def test_shape_preserved(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = RandomTimeShift(max_shift=100, probability=1.0)
        result_wf, result_lb = aug(wf.copy(), lb.copy())
        assert result_wf.shape == wf.shape
        assert result_lb.shape == lb.shape


class TestRandomChannelDrop:
    def test_shape_preserved(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = RandomChannelDrop(probability=1.0)
        result_wf, _ = aug(wf.copy(), lb.copy())
        assert result_wf.shape == wf.shape

    def test_one_channel_zeroed(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        np.random.seed(42)
        aug = RandomChannelDrop(probability=1.0)
        result_wf, _ = aug(wf.copy(), lb.copy())
        zero_channels = sum(np.all(result_wf[ch] == 0) for ch in range(3))
        assert zero_channels >= 1


class TestRandomPolarityFlip:
    def test_shape_preserved(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = RandomPolarityFlip(probability=1.0)
        result_wf, _ = aug(wf.copy(), lb.copy())
        assert result_wf.shape == wf.shape


class TestCompose:
    def test_chain(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = Compose([
            AddGaussianNoise(probability=1.0),
            AmplitudeScale(probability=1.0),
        ])
        result_wf, result_lb = aug(wf.copy(), lb.copy())
        assert result_wf.shape == wf.shape

    def test_empty_compose(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = Compose([])
        result_wf, _ = aug(wf.copy(), lb.copy())
        np.testing.assert_array_equal(result_wf, wf)


class TestGetDefaultAugmentation:
    def test_returns_compose(self):
        aug = get_default_augmentation()
        assert isinstance(aug, Compose)

    def test_callable(self, waveform_and_labels):
        wf, lb = waveform_and_labels
        aug = get_default_augmentation()
        result_wf, result_lb = aug(wf.copy(), lb.copy())
        assert result_wf.shape == wf.shape
