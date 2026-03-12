# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commits
- Do NOT add `Co-Authored-By` trailer to commit messages
- Do NOT include Claude as a contributor in any commit

## Project Overview
TPhaseNet: a U-Net + Transformer hybrid model for seismic P-wave and S-wave phase detection. Trained on the STEAD dataset (~1.27M traces). Input is 3-component seismic waveforms (Z, N, E), output is per-sample [Noise, P, S] probabilities via softmax.

## Common Commands

```bash
# Install
pip install -e .                    # core
pip install -e ".[seisbench]"       # with SeisBench pretrained models

# Run all tests (82 tests)
pytest

# Run specific test file or test
pytest tests/test_models.py
pytest tests/test_models.py -k "TestTPhaseNet"

# Train
python -m scripts.train --csv merged.csv --hdf5 merged.hdf5 --config config/default.yaml

# Inference
python -m scripts.predict --input data.mseed --model checkpoints/best.pt --plot

# Inference with SeisBench model
python -m scripts.predict --input data.mseed --use-seisbench PhaseNet --pretrained stead

# Evaluate on test set
python -m scripts.evaluate --model checkpoints/best.pt --csv merged.csv --hdf5 merged.hdf5
```

## Architecture

**Model** (`models/`): U-Net encoder-decoder with Transformer blocks at encoder levels 3+.
- `tphasenet.py` — top-level model, `TPhaseNet.from_config(cfg)` factory method
- `encoder.py` — 7 downsampling levels, applies TransformerBlock from `transformer_start_level`
- `decoder.py` — mirror upsampling with skip connections, 1x1 conv + softmax output
- `conv_blocks.py` — DownBlock (conv+stride=4 downsample) / UpBlock (ConvTranspose upsample)
- `transformer_block.py` — sinusoidal positional encoding + PyTorch TransformerEncoderLayer

**Data** (`data/`): STEAD HDF5 dataset with preprocessing pipeline.
- `stead_dataset.py` — PyTorch Dataset, deterministic MD5-hash split by `source_id`
- `preprocessing.py` — demean → detrend → bandpass (0.5-45Hz) → normalize (std)
- `augmentation.py` — Compose-able transforms (noise, amplitude scale, time shift, channel drop, polarity flip)
- `label_utils.py` — Gaussian labels at P/S arrivals (sigma=20 samples)

**Training** (`training/`):
- `trainer.py` — training loop with AMP, early stopping, ReduceLROnPlateau
- `losses.py` — WeightedCrossEntropyLoss (default, weights [1, 30, 30]) and FocalCrossEntropyLoss
- `metrics.py` — tolerance-based (±50 samples) precision/recall/F1, pick residuals

**Inference** (`inference/`):
- `picker.py` — `SeismicPicker` class: loads model, handles sliding window for long waveforms
- `postprocessing.py` — peak extraction via `scipy.signal.find_peaks`, S-after-P constraint
- `output_formatter.py` — JSON/CSV output with absolute timestamps
- `seisbench_picker.py` — wrapper for SeisBench pretrained models (PhaseNet, EQTransformer)

**Config** (`config/`):
- `default.yaml` — primary config (model, data, inference params)
- `defaults.py` — Python dict fallback, single source of truth for all scripts

## Key Design Decisions
- All tensor shapes are **(B, C, T)** — batch, channels (3), time (6000 samples = 60s @ 100Hz)
- Train/val/test split is deterministic via MD5 hash of `source_id`, not random seed
- Loss heavily weights P/S classes (30x) over noise to handle class imbalance
- Sliding window inference uses 50% overlap with weighted averaging for merge
- Transformer uses pre-norm (`norm_first=True`), GELU activation, residual connection
