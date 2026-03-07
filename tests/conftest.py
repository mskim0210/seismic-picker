"""Shared fixtures for seismic-picker tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_waveform():
    """3-component waveform (Z, N, E), 60 seconds at 100 Hz."""
    np.random.seed(42)
    return np.random.randn(3, 6000).astype(np.float32)


@pytest.fixture
def short_waveform():
    """Short 3-component waveform for quick model tests."""
    np.random.seed(42)
    return np.random.randn(3, 1024).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Labels with P at sample 2000 and S at sample 4000."""
    from data.label_utils import generate_labels
    return generate_labels(6000, p_sample=2000, s_sample=4000, sigma=20)


@pytest.fixture
def batch_tensor():
    """Batch of 4 waveforms as torch tensor."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 6000)


@pytest.fixture
def model_config():
    """Minimal TPhaseNet config for testing (small model)."""
    return {
        "in_channels": 3,
        "classes": 3,
        "filters_root": 8,
        "depth": 4,
        "kernel_size": 7,
        "stride": 4,
        "transformer_start_level": 2,
        "n_heads": 4,
        "ff_dim_factor": 4,
        "dropout": 0.0,
    }
