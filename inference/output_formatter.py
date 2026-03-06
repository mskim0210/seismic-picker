import json
import csv
import os
from datetime import datetime, timedelta


def _parse_time(start_time_str):
    """ObsPy UTCDateTime 문자열을 datetime으로 파싱."""
    try:
        from obspy import UTCDateTime
        return UTCDateTime(start_time_str).datetime
    except ImportError:
        # ObsPy 없이 기본 파싱
        return datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))


def format_picks_absolute(picks, start_time_str):
    """상대 시각의 pick을 절대 시각으로 변환.

    Args:
        picks: extract_picks에서 반환된 pick 리스트
        start_time_str: 파형 시작 시각 문자열

    Returns:
        list of dict (절대 시각 포함)
    """
    start_dt = _parse_time(start_time_str)
    result = []

    for pick in picks:
        offset = timedelta(seconds=pick["time_offset_sec"])
        abs_time = start_dt + offset

        result.append({
            "phase": pick["phase"],
            "time": abs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "confidence": round(pick["confidence"], 4),
            "uncertainty_sec": round(pick["uncertainty_sec"], 4),
            "sample_index": pick["sample_index"],
        })

    return result


def to_json(picks, metadata, output_path=None):
    """pick 결과를 JSON 형식으로 변환.

    Args:
        picks: 절대 시각이 포함된 pick 리스트
        metadata: mseed 메타데이터 dict
        output_path: 저장 경로 (None이면 dict만 반환)

    Returns:
        dict
    """
    result = {
        "station": metadata.get("station", ""),
        "network": metadata.get("network", ""),
        "location": metadata.get("location", ""),
        "start_time": metadata.get("start_time", ""),
        "channels": metadata.get("channels", []),
        "picks": picks,
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def to_csv(picks_list, output_path):
    """여러 파일의 pick 결과를 CSV로 저장.

    Args:
        picks_list: list of (metadata, picks) tuples
        output_path: CSV 저장 경로
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "network", "station", "location", "phase",
        "time", "confidence", "uncertainty_sec",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metadata, picks in picks_list:
            for pick in picks:
                writer.writerow({
                    "network": metadata.get("network", ""),
                    "station": metadata.get("station", ""),
                    "location": metadata.get("location", ""),
                    "phase": pick["phase"],
                    "time": pick["time"],
                    "confidence": round(pick["confidence"], 4),
                    "uncertainty_sec": round(pick["uncertainty_sec"], 4),
                })
