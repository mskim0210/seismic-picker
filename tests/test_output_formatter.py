"""Tests for inference/output_formatter.py"""

import json
import os
import tempfile
import pytest
from inference.output_formatter import format_picks_absolute, to_json, to_csv


@pytest.fixture
def sample_picks():
    return [
        {
            "phase": "P",
            "sample_index": 2000,
            "time_offset_sec": 20.0,
            "confidence": 0.95,
            "uncertainty_sec": 0.05,
        },
        {
            "phase": "S",
            "sample_index": 4000,
            "time_offset_sec": 40.0,
            "confidence": 0.88,
            "uncertainty_sec": 0.08,
        },
    ]


@pytest.fixture
def sample_metadata():
    return {
        "station": "TEST",
        "network": "KG",
        "location": "00",
        "start_time": "2024-01-15T00:00:00.000000Z",
        "channels": ["HHZ", "HHN", "HHE"],
    }


class TestFormatPicksAbsolute:
    def test_adds_absolute_time(self, sample_picks):
        result = format_picks_absolute(sample_picks, "2024-01-15T00:00:00.000000Z")
        assert len(result) == 2
        for pick in result:
            assert "time" in pick

    def test_preserves_fields(self, sample_picks):
        result = format_picks_absolute(sample_picks, "2024-01-15T00:00:00.000000Z")
        assert result[0]["phase"] == "P"
        assert result[0]["confidence"] == 0.95
        assert result[1]["phase"] == "S"


class TestToJson:
    def test_structure(self, sample_picks, sample_metadata):
        result = to_json(sample_picks, sample_metadata)
        assert "station" in result
        assert "picks" in result
        assert result["station"] == "TEST"

    def test_write_file(self, sample_picks, sample_metadata):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            to_json(sample_picks, sample_metadata, output_path=path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "picks" in data
        finally:
            os.unlink(path)


class TestToCsv:
    def test_write_csv(self, sample_picks, sample_metadata):
        # to_csv expects list of (metadata, picks) tuples
        # picks need "time" field (absolute time)
        picks_with_time = [
            {**p, "time": "2024-01-15T00:00:20.000"} for p in sample_picks
        ]
        picks_list = [(sample_metadata, picks_with_time)]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(picks_list, path)
            assert os.path.exists(path)
            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) >= 2  # header + at least 1 data row
        finally:
            os.unlink(path)
