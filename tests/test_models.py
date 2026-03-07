"""Tests for models/ (conv_blocks, transformer_block, encoder, decoder, tphasenet)."""

import torch
import pytest
from models.conv_blocks import DownBlock, UpBlock
from models.transformer_block import SinusoidalPositionalEncoding, TransformerBlock
from models.encoder import Encoder
from models.decoder import Decoder
from models.tphasenet import TPhaseNet


class TestDownBlock:
    def test_output_shapes(self):
        block = DownBlock(in_channels=8, out_channels=16, kernel_size=7, stride=4)
        x = torch.randn(2, 8, 256)
        skip, out = block(x)
        assert skip.shape == (2, 16, 256)
        assert out.shape == (2, 16, 64)  # 256 / 4

    def test_different_stride(self):
        block = DownBlock(in_channels=8, out_channels=16, kernel_size=7, stride=2)
        x = torch.randn(2, 8, 256)
        skip, out = block(x)
        assert out.shape[2] == 128  # 256 / 2


class TestUpBlock:
    def test_output_shapes(self):
        block = UpBlock(in_channels=16, out_channels=8, kernel_size=7, stride=4)
        x = torch.randn(2, 16, 64)
        skip = torch.randn(2, 8, 256)
        out = block(x, skip)
        assert out.shape == (2, 8, 256)


class TestSinusoidalPositionalEncoding:
    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=1000)
        x = torch.randn(2, 500, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_adds_positional_info(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=1000)
        x = torch.zeros(1, 100, 64)
        out = pe(x)
        assert not torch.allclose(out, x)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4, dropout=0.0)
        x = torch.randn(2, 64, 100)  # (B, C, T)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = TransformerBlock(d_model=64, n_heads=4, dropout=0.0)
        block.eval()
        x = torch.randn(2, 64, 50)
        out = block(x)
        # Output should not be identical (transformer modifies) but same shape
        assert out.shape == x.shape


class TestEncoder:
    def test_output_shapes(self, model_config):
        enc = Encoder(
            in_channels=model_config["in_channels"],
            filters_root=model_config["filters_root"],
            depth=model_config["depth"],
            kernel_size=model_config["kernel_size"],
            stride=model_config["stride"],
            transformer_start_level=model_config["transformer_start_level"],
        )
        x = torch.randn(2, 3, 1024)
        bottleneck, skips = enc(x)
        # Should have depth-1 skip connections
        assert len(skips) == model_config["depth"] - 1
        assert bottleneck.dim() == 3

    def test_downsampling(self, model_config):
        enc = Encoder(
            in_channels=3,
            filters_root=8,
            depth=4,
            stride=4,
        )
        x = torch.randn(2, 3, 1024)
        bottleneck, skips = enc(x)
        # After 3 downsampling stages (depth-1), T should shrink
        assert bottleneck.shape[2] < x.shape[2]


class TestDecoder:
    def test_output_shape(self, model_config):
        enc = Encoder(
            in_channels=model_config["in_channels"],
            filters_root=model_config["filters_root"],
            depth=model_config["depth"],
            stride=model_config["stride"],
        )
        dec = Decoder(
            classes=model_config["classes"],
            filters_root=model_config["filters_root"],
            depth=model_config["depth"],
            stride=model_config["stride"],
        )
        x = torch.randn(2, 3, 1024)
        bottleneck, skips = enc(x)
        out = dec(bottleneck, skips)
        assert out.shape == (2, 3, 1024)


class TestTPhaseNet:
    def test_forward_shape(self, model_config):
        model = TPhaseNet(**model_config)
        model.eval()
        x = torch.randn(1, 3, 1024)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 1024)

    def test_output_softmax(self, model_config):
        model = TPhaseNet(**model_config)
        model.eval()
        x = torch.randn(1, 3, 1024)
        with torch.no_grad():
            out = model(x)
        # Softmax over classes (dim=1): sum should be ~1
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_output_range(self, model_config):
        model = TPhaseNet(**model_config)
        model.eval()
        x = torch.randn(1, 3, 1024)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_processing(self, model_config):
        model = TPhaseNet(**model_config)
        model.eval()
        x = torch.randn(4, 3, 1024)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3, 1024)

    def test_from_config(self, model_config):
        wrapped_config = {"model": model_config}
        model = TPhaseNet.from_config(wrapped_config)
        assert isinstance(model, TPhaseNet)

    def test_count_parameters(self, model_config):
        model = TPhaseNet(**model_config)
        count = model.count_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_gradient_flow(self, model_config):
        model = TPhaseNet(**model_config)
        model.train()
        x = torch.randn(2, 3, 1024, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
