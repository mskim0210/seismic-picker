"""Tests for training/losses.py"""

import torch
import pytest
from training.losses import WeightedCrossEntropyLoss, FocalCrossEntropyLoss


@pytest.fixture
def pred_and_target():
    """Simulated prediction (softmax output) and one-hot target."""
    torch.manual_seed(42)
    # Predictions: softmax-like (sum to 1 along class dim)
    logits = torch.randn(4, 3, 1000)
    pred = torch.softmax(logits, dim=1)
    # Target: one-hot labels
    target = torch.zeros(4, 3, 1000)
    target[:, 0, :] = 1.0  # all noise
    target[:, 1, 500] = 1.0  # P pick
    target[:, 0, 500] = 0.0
    target[:, 2, 800] = 1.0  # S pick
    target[:, 0, 800] = 0.0
    return pred, target


class TestWeightedCrossEntropyLoss:
    def test_output_scalar(self, pred_and_target):
        pred, target = pred_and_target
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # scalar

    def test_positive_loss(self, pred_and_target):
        pred, target = pred_and_target
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        target = torch.zeros(2, 3, 100)
        target[:, 0, :] = 1.0
        pred = target.clone()
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_custom_weights(self, pred_and_target):
        pred, target = pred_and_target
        loss_fn = WeightedCrossEntropyLoss(class_weights=[1.0, 50.0, 50.0])
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_gradient_flows(self, pred_and_target):
        pred, target = pred_and_target
        pred.requires_grad_(True)
        loss_fn = WeightedCrossEntropyLoss()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestFocalCrossEntropyLoss:
    def test_output_scalar(self, pred_and_target):
        pred, target = pred_and_target
        loss_fn = FocalCrossEntropyLoss()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_positive_loss(self, pred_and_target):
        pred, target = pred_and_target
        loss_fn = FocalCrossEntropyLoss()
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_gamma_effect(self, pred_and_target):
        pred, target = pred_and_target
        loss_g0 = FocalCrossEntropyLoss(gamma=0.0)(pred, target).item()
        loss_g2 = FocalCrossEntropyLoss(gamma=2.0)(pred, target).item()
        # Higher gamma should reduce loss for easy samples
        assert loss_g2 < loss_g0

    def test_gradient_flows(self, pred_and_target):
        pred, target = pred_and_target
        pred.requires_grad_(True)
        loss_fn = FocalCrossEntropyLoss()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
