"""SeisBench 사전학습 모델을 활용한 지진파 위상 검출 및 picking.

SeisBench의 PhaseNet 또는 EQTransformer 사전학습 가중치를 사용하여
추가 학습 없이 바로 mseed 파일에서 P/S파를 검출.

사용법:
    picker = SeisBenchPicker(model_name="PhaseNet", pretrained="stead")
    result = picker.pick("station.mseed")
"""

import numpy as np
from pathlib import Path

try:
    import seisbench.models as sbm
    from obspy import read as obspy_read
    HAS_SEISBENCH = True
except ImportError:
    HAS_SEISBENCH = False

from .output_formatter import format_picks_absolute, to_json


# SeisBench 모델 매핑
MODEL_MAP = {
    "PhaseNet": "PhaseNet",
    "phasenet": "PhaseNet",
    "EQTransformer": "EQTransformer",
    "eqtransformer": "EQTransformer",
    "EQT": "EQTransformer",
}

# 사용 가능한 사전학습 가중치
AVAILABLE_WEIGHTS = {
    "PhaseNet": [
        "stead", "ethz", "instance", "geofon",
        "iquique", "lendb", "neic", "scedc", "original",
    ],
    "EQTransformer": [
        "stead", "ethz", "instance", "geofon",
        "iquique", "lendb", "neic", "scedc", "original",
    ],
}


class SeisBenchPicker:
    """SeisBench 사전학습 모델 기반 picker.

    Args:
        model_name: 'PhaseNet' 또는 'EQTransformer'
        pretrained: 사전학습 가중치 이름 (예: 'stead', 'instance', 'original')
        device: 'cuda', 'cpu', 'mps'
        p_threshold: P파 검출 임계값
        s_threshold: S파 검출 임계값
    """

    def __init__(self, model_name="PhaseNet", pretrained="stead",
                 device="cpu", p_threshold=0.3, s_threshold=0.3):
        if not HAS_SEISBENCH:
            raise ImportError(
                "SeisBench가 설치되지 않았습니다. "
                "pip install seisbench 로 설치해주세요."
            )

        model_key = MODEL_MAP.get(model_name, model_name)

        model_class = getattr(sbm, model_key, None)
        if model_class is None:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_MAP.keys())}"
            )

        self.model = model_class.from_pretrained(pretrained)
        self.model.eval()

        self.model_name = model_key
        self.p_threshold = p_threshold
        self.s_threshold = s_threshold

    def pick(self, mseed_path):
        """단일 mseed 파일에서 위상 pick 추출.

        Args:
            mseed_path: mseed 파일 경로

        Returns:
            dict: pick 결과
        """
        st = obspy_read(str(mseed_path))

        # SeisBench annotate 사용
        annotations = self.model.annotate(st)

        # 메타데이터 추출
        ref_trace = st[0]
        metadata = {
            "start_time": str(ref_trace.stats.starttime),
            "station": ref_trace.stats.station,
            "network": ref_trace.stats.network,
            "location": ref_trace.stats.location,
            "channels": [tr.stats.channel for tr in st],
        }

        # SeisBench classify 사용하여 pick 추출
        classify_output = self.model.classify(
            st,
            P_threshold=self.p_threshold,
            S_threshold=self.s_threshold,
        )

        # SeisBench >= 0.11: classify_output.picks로 접근
        if hasattr(classify_output, 'picks'):
            pick_list = classify_output.picks
        else:
            pick_list = classify_output

        # Pick 결과 포맷팅
        formatted_picks = []
        for pick in pick_list:
            formatted_picks.append({
                "phase": pick.phase,
                "time": str(pick.peak_time),
                "confidence": round(float(pick.peak_value), 4),
                "uncertainty_sec": 0.0,
                "sample_index": 0,
            })

        return to_json(formatted_picks, metadata)

    def pick_batch(self, mseed_paths, output_dir=None):
        """여러 mseed 파일 일괄 처리."""
        results = []
        for path in mseed_paths:
            try:
                result = self.pick(path)
                results.append(result)

                if output_dir:
                    fname = Path(path).stem + "_picks.json"
                    out_path = str(Path(output_dir) / fname)
                    to_json(result.get("picks", []), {
                        "station": result.get("station", ""),
                        "network": result.get("network", ""),
                        "location": result.get("location", ""),
                        "start_time": result.get("start_time", ""),
                        "channels": result.get("channels", []),
                    }, out_path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({"error": str(e), "file": str(path)})

        return results

    def get_probabilities(self, mseed_path):
        """확률 곡선 반환 (시각화용).

        Returns:
            annotations: ObsPy Stream (확률 곡선)
            original: ObsPy Stream (원본 파형)
        """
        st = obspy_read(str(mseed_path))
        annotations = self.model.annotate(st)
        return annotations, st

    @staticmethod
    def list_models():
        """사용 가능한 모델과 가중치 목록 출력."""
        if not HAS_SEISBENCH:
            print("SeisBench가 설치되지 않았습니다.")
            return

        for model_name in ["PhaseNet", "EQTransformer"]:
            model_class = getattr(sbm, model_name)
            print(f"\n{model_name}:")
            try:
                pretrained = model_class.list_pretrained()
                for name in pretrained:
                    print(f"  - {name}")
            except Exception as e:
                print(f"  (목록 조회 실패: {e})")
                print(f"  알려진 가중치: {AVAILABLE_WEIGHTS.get(model_name, [])}")
