import os
import numpy as np
import torch
import yaml
from pathlib import Path

from ..models.tphasenet import TPhaseNet
from ..data.mseed_loader import load_mseed, load_mseed_stream
from ..config.defaults import get_default_config
from .postprocessing import extract_picks, merge_sliding_window_probs
from .output_formatter import format_picks_absolute, to_json


class SeismicPicker:
    """End-to-end 지진파 위상 검출 및 picking 파이프라인.

    mseed 파일을 입력받아 P파/S파 도착 시각과 신뢰도를 반환.
    """

    def __init__(self, model_path, config_path=None, device=None):
        """
        Args:
            model_path: 학습된 모델 체크포인트 경로 (.pt)
            config_path: 설정 YAML 경로 (None이면 기본값 사용)
            device: 'cuda', 'cpu', 또는 'mps' (None이면 자동 선택)
        """
        # 설정 로드
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = get_default_config()

        # 디바이스 설정
        if device is None:
            device = self.config.get("inference", {}).get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # 모델 로드
        self.model = TPhaseNet.from_config(self.config)
        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=True)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # 추론 파라미터
        inf_cfg = self.config.get("inference", {})
        self.peak_cfg = inf_cfg.get("peak_detection", {})
        self.window_size = inf_cfg.get("sliding_window", {}).get("window_size", 6000)
        self.step = inf_cfg.get("sliding_window", {}).get("step", 3000)
        self.target_length = self.config.get("data", {}).get("target_length", 6000)
        self.sampling_rate = self.config.get("data", {}).get("sampling_rate", 100.0)

    def pick(self, mseed_path):
        """단일 mseed 파일에서 위상 pick 추출.

        Args:
            mseed_path: mseed 파일 경로

        Returns:
            dict: {
                'station': str,
                'network': str,
                'picks': list of pick dicts,
                ...
            }
        """
        data_cfg = self.config.get("data", {})

        # 전체 길이 파형 로드
        waveform, metadata = load_mseed_stream(
            mseed_path,
            target_sampling_rate=self.sampling_rate,
            config=data_cfg,
        )

        n_samples = waveform.shape[1]

        if n_samples <= self.window_size:
            # 짧은 파형: zero-pad하여 바로 추론
            prob_curves = self._infer_single(waveform, self.target_length)
            # pad한 경우 원래 길이만큼만 사용
            prob_curves = prob_curves[:, :n_samples]
        else:
            # 긴 파형: 슬라이딩 윈도우
            prob_curves = self._infer_sliding(waveform)

        # Pick 추출
        picks = extract_picks(
            prob_curves,
            sampling_rate=self.sampling_rate,
            min_height=self.peak_cfg.get("min_height", 0.3),
            min_distance=self.peak_cfg.get("min_distance", 100),
            min_prominence=self.peak_cfg.get("min_prominence", 0.1),
        )

        # 절대 시각 변환
        picks = format_picks_absolute(picks, metadata["start_time"])

        return to_json(picks, metadata)

    def pick_batch(self, mseed_paths, output_dir=None, output_format="json"):
        """여러 mseed 파일 일괄 처리.

        Args:
            mseed_paths: mseed 파일 경로 리스트
            output_dir: 결과 저장 디렉토리 (None이면 저장 안함)
            output_format: 'json' or 'csv'

        Returns:
            list of pick result dicts
        """
        results = []

        for path in mseed_paths:
            try:
                result = self.pick(path)
                results.append(result)

                if output_dir is not None and output_format == "json":
                    fname = Path(path).stem + "_picks.json"
                    out_path = os.path.join(output_dir, fname)
                    to_json(result["picks"], {
                        "station": result["station"],
                        "network": result["network"],
                        "location": result.get("location", ""),
                        "start_time": result["start_time"],
                        "channels": result.get("channels", []),
                    }, out_path)

            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({"error": str(e), "file": str(path)})

        return results

    def get_probabilities(self, mseed_path):
        """확률 곡선 반환 (시각화용).

        Args:
            mseed_path: mseed 파일 경로

        Returns:
            prob_curves: (3, N) numpy array [Noise, P, S]
            metadata: dict
        """
        data_cfg = self.config.get("data", {})
        waveform, metadata = load_mseed_stream(
            mseed_path,
            target_sampling_rate=self.sampling_rate,
            config=data_cfg,
        )

        n_samples = waveform.shape[1]

        if n_samples <= self.window_size:
            prob_curves = self._infer_single(waveform, self.target_length)
            prob_curves = prob_curves[:, :n_samples]
        else:
            prob_curves = self._infer_sliding(waveform)

        return prob_curves, waveform, metadata

    def _infer_single(self, waveform, target_length):
        """단일 윈도우 추론.

        Args:
            waveform: (3, N) numpy array, N <= target_length
            target_length: 모델 입력 길이

        Returns:
            (3, target_length) numpy array
        """
        n = waveform.shape[1]
        if n < target_length:
            padded = np.zeros((3, target_length), dtype=np.float32)
            padded[:, :n] = waveform
            waveform = padded

        tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        return output.squeeze(0).cpu().numpy()

    def _infer_sliding(self, waveform):
        """슬라이딩 윈도우 추론.

        Args:
            waveform: (3, N) numpy array, N > window_size

        Returns:
            (3, N) numpy array
        """
        n_samples = waveform.shape[1]
        prob_list = []

        start = 0
        while start < n_samples:
            end = min(start + self.window_size, n_samples)
            window = waveform[:, start:end]

            # 마지막 윈도우가 짧으면 zero-pad
            if window.shape[1] < self.window_size:
                padded = np.zeros((3, self.window_size), dtype=np.float32)
                padded[:, :window.shape[1]] = window
                window = padded

            tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)

            probs = output.squeeze(0).cpu().numpy()

            # 마지막 윈도우는 실제 길이만큼만 사용
            actual_len = end - start
            prob_list.append(probs[:, :actual_len])

            if end >= n_samples:
                break
            start += self.step

        return merge_sliding_window_probs(
            prob_list, self.window_size, self.step, n_samples
        )

