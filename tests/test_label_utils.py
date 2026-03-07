"""Tests for data/label_utils.py"""

import numpy as np
import pytest
from data.label_utils import generate_gaussian_label, generate_labels


class TestGenerateGaussianLabel:
    def test_peak_location(self):
        label = generate_gaussian_label(6000, arrival_sample=3000, sigma=20)
        assert np.argmax(label) == 3000

    def test_peak_value(self):
        label = generate_gaussian_label(6000, arrival_sample=3000, sigma=20)
        assert abs(label[3000] - 1.0) < 1e-5

    def test_shape(self):
        label = generate_gaussian_label(6000, arrival_sample=1000, sigma=20)
        assert label.shape == (6000,)
        assert label.dtype == np.float32

    def test_tails_near_zero(self):
        label = generate_gaussian_label(6000, arrival_sample=3000, sigma=20)
        assert label[0] < 1e-10
        assert label[5999] < 1e-10

    def test_different_sigma(self):
        narrow = generate_gaussian_label(6000, 3000, sigma=10)
        wide = generate_gaussian_label(6000, 3000, sigma=40)
        # Wider sigma -> more spread -> higher value far from peak
        assert wide[3050] > narrow[3050]

    def test_none_arrival(self):
        label = generate_gaussian_label(6000, arrival_sample=None, sigma=20)
        assert np.all(label == 0.0)


class TestGenerateLabels:
    def test_shape(self):
        labels = generate_labels(6000, p_sample=2000, s_sample=4000)
        assert labels.shape == (3, 6000)

    def test_three_channels(self):
        labels = generate_labels(6000, p_sample=2000, s_sample=4000)
        # Channel 0: Noise, 1: P, 2: S
        assert np.argmax(labels[1]) == 2000
        assert np.argmax(labels[2]) == 4000

    def test_noise_complement(self):
        labels = generate_labels(6000, p_sample=2000, s_sample=4000)
        # Noise = 1 - P - S, clipped
        total = labels.sum(axis=0)
        assert np.allclose(total, 1.0, atol=0.01)

    def test_no_arrivals(self):
        labels = generate_labels(6000, p_sample=None, s_sample=None)
        assert np.allclose(labels[0], 1.0)
        assert np.allclose(labels[1], 0.0)
        assert np.allclose(labels[2], 0.0)

    def test_only_p(self):
        labels = generate_labels(6000, p_sample=2000, s_sample=None)
        assert np.argmax(labels[1]) == 2000
        assert np.allclose(labels[2], 0.0)
