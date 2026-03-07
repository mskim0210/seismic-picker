"""Tests for training/metrics.py"""

import numpy as np
import torch
import pytest
from training.metrics import compute_pick_metrics
from data.label_utils import generate_labels


class TestComputePickMetrics:
    def _make_batch(self, p_sample, s_sample, n_samples=6000, batch_size=2):
        """Create matching pred/true with peaks at given samples."""
        labels = generate_labels(n_samples, p_sample, s_sample, sigma=20)
        batch = np.stack([labels] * batch_size)
        return torch.from_numpy(batch).float()

    def test_perfect_prediction(self):
        true = self._make_batch(2000, 4000)
        pred = true.clone()
        metrics = compute_pick_metrics(pred, true)
        assert metrics["p_precision"] == 1.0
        assert metrics["p_recall"] == 1.0
        assert metrics["p_f1"] == 1.0
        assert metrics["s_precision"] == 1.0
        assert metrics["s_recall"] == 1.0
        assert metrics["s_f1"] == 1.0

    def test_no_picks(self):
        true = self._make_batch(2000, 4000)
        # Prediction: all noise
        pred = torch.zeros_like(true)
        pred[:, 0, :] = 1.0
        metrics = compute_pick_metrics(pred, true)
        assert metrics["p_recall"] == 0.0
        assert metrics["s_recall"] == 0.0

    def test_return_keys(self):
        true = self._make_batch(2000, 4000)
        pred = true.clone()
        metrics = compute_pick_metrics(pred, true)
        expected_keys = [
            "p_precision", "p_recall", "p_f1",
            "s_precision", "s_recall", "s_f1",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_tolerance(self):
        true = self._make_batch(2000, 4000)
        # Shift P pick by 30 samples
        pred_labels = generate_labels(6000, 2030, 4000, sigma=20)
        pred = torch.from_numpy(np.stack([pred_labels, pred_labels])).float()
        # With tolerance=50, should still match
        metrics_wide = compute_pick_metrics(pred, true, tolerance_samples=50)
        assert metrics_wide["p_tp"] > 0
        # With tolerance=10, should NOT match
        metrics_tight = compute_pick_metrics(pred, true, tolerance_samples=10)
        assert metrics_tight["p_tp"] == 0
