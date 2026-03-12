"""STEAD (STanford EArthquake Dataset) PyTorch Dataset.

STEAD: 1,265,657개 3성분 지진파형, 60초, 100Hz, HDF5+CSV 형식.

다운로드: https://github.com/smousavi05/STEAD
"""

import hashlib
import logging
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

from .label_utils import generate_labels


class STEADDataset(Dataset):
    """STEAD HDF5 데이터셋 로더.

    Args:
        csv_path: merged.csv 경로
        hdf5_path: merged.hdf5 경로
        split: 'train', 'val', 'test'
        target_length: 목표 샘플 수 (6000)
        sigma: Gaussian 라벨 너비 (samples)
        transform: 데이터 증강 함수 (waveform, labels) -> (waveform, labels)
        max_samples: 최대 샘플 수 제한 (디버깅용, None이면 전체)
    """

    def __init__(self, csv_path, hdf5_path, split="train",
                 target_length=6000, sigma=20, transform=None,
                 max_samples=None):
        self.hdf5_path = hdf5_path
        self.target_length = target_length
        self.sigma = sigma
        self.transform = transform

        # CSV 메타데이터 로드
        df = pd.read_csv(csv_path, low_memory=False)

        # split 분할 (source_id 기반 해시 분할)
        df = self._split_data(df, split)

        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)

        self.trace_names = df["trace_name"].values
        self.p_arrivals = df["p_arrival_sample"].values.astype(np.float64)
        self.s_arrivals = df["s_arrival_sample"].values.astype(np.float64)
        self.trace_categories = df["trace_category"].values

        # HDF5 파일은 worker별로 별도 오픈 (multiprocessing 안전)
        self._hdf5_file = None

    def _split_data(self, df, split):
        """source_id 기반 데이터 분할 (데이터 누출 방지)."""
        if "split" in df.columns:
            return df[df["split"] == split].reset_index(drop=True)

        # source_id 기반 해시, NaN인 경우 trace_name으로 fallback
        # hashlib.md5 사용으로 PYTHONHASHSEED에 무관하게 재현 가능
        def _deterministic_hash(value):
            return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % 100

        if "source_id" in df.columns:
            hash_vals = df.apply(
                lambda row: _deterministic_hash(row["source_id"])
                if pd.notna(row["source_id"])
                else _deterministic_hash(row["trace_name"]),
                axis=1,
            )
        else:
            hash_vals = df["trace_name"].apply(_deterministic_hash)

        if split == "train":
            mask = hash_vals < 80
        elif split == "val":
            mask = (hash_vals >= 80) & (hash_vals < 90)
        else:  # test
            mask = hash_vals >= 90

        return df[mask].reset_index(drop=True)

    def _open_hdf5(self):
        """HDF5 파일 오픈 (worker별 독립)."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        self._open_hdf5()

        trace_name = self.trace_names[idx]

        # HDF5에서 파형 읽기: shape (6000, 3) -> (3, 6000)
        # STEAD 형식: columns = [E, N, Z]
        try:
            waveform = np.array(
                self._hdf5_file["data"][trace_name], dtype=np.float32
            )
        except KeyError:
            # earthquake/noise 그룹에서 시도
            for group in ["earthquake/local", "non_earthquake/noise"]:
                key = f"{group}/{trace_name}"
                if key in self._hdf5_file:
                    waveform = np.array(
                        self._hdf5_file[key], dtype=np.float32
                    )
                    break
            else:
                # 빈 파형 반환
                logger.warning(f"Trace not found in HDF5: {trace_name}")
                waveform = np.zeros((self.target_length, 3), dtype=np.float32)

        # (N, 3) -> (3, N), 채널 순서: E,N,Z -> Z,N,E (인덱스 2,1,0)
        if waveform.shape[0] == 3:
            # 이미 (3, N) 형식
            pass
        else:
            waveform = waveform.T  # (3, N)

        # 채널 재정렬: STEAD는 E(0), N(1), Z(2) -> Z(0), N(1), E(2)
        if waveform.shape[0] == 3:
            waveform = waveform[[2, 1, 0], :]

        # 길이 조정
        n = waveform.shape[1]
        if n < self.target_length:
            padded = np.zeros((3, self.target_length), dtype=np.float32)
            padded[:, :n] = waveform
            waveform = padded
        elif n > self.target_length:
            waveform = waveform[:, :self.target_length]

        # 정규화
        std = np.std(waveform)
        if std > 1e-8:
            waveform = waveform / std

        # 라벨 생성
        p_sample = self.p_arrivals[idx]
        s_sample = self.s_arrivals[idx]
        labels = generate_labels(self.target_length, p_sample, s_sample,
                                 self.sigma)

        # 데이터 증강
        if self.transform is not None:
            waveform, labels = self.transform(waveform, labels)

        return (
            torch.from_numpy(waveform),
            torch.from_numpy(labels),
        )

    def __del__(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()


def worker_init_fn(worker_id):
    """DataLoader worker 초기화 함수.

    각 worker에서 HDF5 파일을 독립적으로 열도록 보장.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._hdf5_file = None  # 각 worker에서 새로 열기
