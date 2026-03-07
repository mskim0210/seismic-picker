"""Tests for inference/postprocessing.py"""

import numpy as np
import pytest
from inference.postprocessing import extract_picks, merge_sliding_window_probs
from data.label_utils import generate_labels


class TestExtractPicks:
    def test_detects_p_and_s(self):
        prob = generate_labels(6000, p_sample=2000, s_sample=4000, sigma=20)
        picks = extract_picks(prob, sampling_rate=100.0)
        phases = [p["phase"] for p in picks]
        assert "P" in phases
        assert "S" in phases

    def test_pick_location_accuracy(self):
        prob = generate_labels(6000, p_sample=2000, s_sample=4000, sigma=20)
        picks = extract_picks(prob, sampling_rate=100.0)
        p_picks = [p for p in picks if p["phase"] == "P"]
        s_picks = [p for p in picks if p["phase"] == "S"]
        assert len(p_picks) == 1
        assert len(s_picks) == 1
        assert abs(p_picks[0]["sample_index"] - 2000) < 5
        assert abs(s_picks[0]["sample_index"] - 4000) < 5

    def test_confidence_range(self):
        prob = generate_labels(6000, p_sample=2000, s_sample=4000, sigma=20)
        picks = extract_picks(prob, sampling_rate=100.0)
        for pick in picks:
            assert 0.0 <= pick["confidence"] <= 1.0

    def test_no_picks_for_noise(self):
        prob = np.zeros((3, 6000), dtype=np.float32)
        prob[0, :] = 1.0  # all noise
        picks = extract_picks(prob, sampling_rate=100.0)
        assert len(picks) == 0

    def test_time_offset(self):
        prob = generate_labels(6000, p_sample=2000, s_sample=4000, sigma=20)
        picks = extract_picks(prob, sampling_rate=100.0)
        p_pick = [p for p in picks if p["phase"] == "P"][0]
        expected_time = 2000 / 100.0  # 20.0 sec
        assert abs(p_pick["time_offset_sec"] - expected_time) < 0.1

    def test_threshold_filtering(self):
        prob = generate_labels(6000, p_sample=2000, s_sample=4000, sigma=20)
        # Scale down so peaks are below threshold
        prob[1] *= 0.2
        prob[2] *= 0.2
        prob[0] = np.clip(1.0 - prob[1] - prob[2], 0, 1)
        picks = extract_picks(prob, sampling_rate=100.0, min_height=0.3)
        assert len(picks) == 0


class TestMergeSlidingWindowProbs:
    def test_output_shape(self):
        window_size = 6000
        step = 3000
        total_length = 15000
        n_windows = (total_length - window_size) // step + 1
        prob_list = [np.random.rand(3, window_size).astype(np.float32)
                     for _ in range(n_windows)]
        merged = merge_sliding_window_probs(prob_list, window_size, step, total_length)
        assert merged.shape == (3, total_length)

    def test_single_window(self):
        prob = np.random.rand(3, 6000).astype(np.float32)
        merged = merge_sliding_window_probs([prob], 6000, 3000, 6000)
        np.testing.assert_array_almost_equal(merged, prob)

    def test_overlap_averaging(self):
        window_size = 100
        step = 50
        total_length = 150
        prob1 = np.ones((3, 100), dtype=np.float32) * 0.2
        prob2 = np.ones((3, 100), dtype=np.float32) * 0.8
        merged = merge_sliding_window_probs([prob1, prob2], window_size, step, total_length)
        # Overlap region (50-99) should be averaged: (0.2 + 0.8) / 2 = 0.5
        assert abs(merged[0, 75] - 0.5) < 0.01
