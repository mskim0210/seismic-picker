"""Tests for data/preprocessing.py"""

import numpy as np
import pytest
from data.preprocessing import demean, detrend, bandpass_filter, normalize, preprocess


class TestDemean:
    def test_zero_mean(self, sample_waveform):
        result = demean(sample_waveform)
        for ch in range(3):
            assert abs(result[ch].mean()) < 1e-6

    def test_shape_preserved(self, sample_waveform):
        result = demean(sample_waveform)
        assert result.shape == sample_waveform.shape

    def test_already_zero_mean(self):
        data = np.array([[1, -1, 1, -1]], dtype=np.float32)
        result = demean(data)
        assert abs(result.mean()) < 1e-6


class TestDetrend:
    def test_removes_linear_trend(self):
        t = np.arange(1000, dtype=np.float32)
        data = np.stack([t * 2 + 5, t * -1 + 3, t * 0.5], axis=0)
        result = detrend(data)
        for ch in range(3):
            coeffs = np.polyfit(np.arange(1000), result[ch], 1)
            assert abs(coeffs[0]) < 1e-3  # slope near zero

    def test_shape_preserved(self, sample_waveform):
        result = detrend(sample_waveform)
        assert result.shape == sample_waveform.shape


class TestBandpassFilter:
    def test_shape_preserved(self, sample_waveform):
        result = bandpass_filter(sample_waveform)
        assert result.shape == sample_waveform.shape

    def test_removes_dc(self):
        data = np.ones((3, 6000), dtype=np.float32) * 10.0
        result = bandpass_filter(data, freq_min=0.5, freq_max=45.0, sampling_rate=100.0)
        assert np.abs(result).max() < 1.0  # DC should be filtered out

    def test_custom_params(self, sample_waveform):
        result = bandpass_filter(sample_waveform, freq_min=1.0, freq_max=20.0, sampling_rate=100.0)
        assert result.shape == sample_waveform.shape


class TestNormalize:
    def test_std_normalization(self, sample_waveform):
        result = normalize(sample_waveform, method="std")
        for ch in range(3):
            assert abs(result[ch].std() - 1.0) < 0.1

    def test_peak_normalization(self, sample_waveform):
        result = normalize(sample_waveform, method="peak")
        for ch in range(3):
            assert np.abs(result[ch]).max() <= 1.0 + 1e-6

    def test_minmax_normalization(self, sample_waveform):
        result = normalize(sample_waveform, method="minmax")
        for ch in range(3):
            assert result[ch].min() >= -1e-6
            assert result[ch].max() <= 1.0 + 1e-6

    def test_zero_input(self):
        data = np.zeros((3, 100), dtype=np.float32)
        result = normalize(data, method="std")
        assert not np.any(np.isnan(result))

    def test_shape_preserved(self, sample_waveform):
        result = normalize(sample_waveform, method="std")
        assert result.shape == sample_waveform.shape


class TestPreprocess:
    def test_full_pipeline(self, sample_waveform):
        result = preprocess(sample_waveform, sampling_rate=100.0)
        assert result.shape == sample_waveform.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
