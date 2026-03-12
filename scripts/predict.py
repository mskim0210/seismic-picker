#!/usr/bin/env python3
"""지진파 위상 검출 및 picking CLI.

사용법:
    # TPhaseNet 모델로 추론
    python -m scripts.predict --input station.mseed --model checkpoint.pt

    # SeisBench 사전학습 모델로 추론 (학습 필요 없음)
    python -m scripts.predict --input station.mseed --use-seisbench PhaseNet --pretrained stead

    # 디렉토리 일괄 처리
    python -m scripts.predict --input-dir ./data/ --model checkpoint.pt --output-dir ./results/

    # 옵션
    python -m scripts.predict --input station.mseed --model checkpoint.pt \\
        --config config/default.yaml --format json --device cuda --threshold 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from glob import glob

# 프로젝트 루트를 path에 추가
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.picker import SeismicPicker
from inference.output_formatter import to_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="TPhaseNet 지진파 위상 검출 및 Picking"
    )

    # 입력
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i", type=str,
        help="단일 mseed 파일 경로"
    )
    input_group.add_argument(
        "--input-dir", type=str,
        help="mseed 파일들이 있는 디렉토리 경로"
    )

    # 모델
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="학습된 모델 체크포인트 경로 (.pt)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="설정 YAML 경로 (기본: 내장 기본값)"
    )

    # SeisBench 모드
    parser.add_argument(
        "--use-seisbench", type=str, default=None,
        choices=["PhaseNet", "EQTransformer"],
        help="SeisBench 사전학습 모델 사용 (모델 학습 불필요)"
    )
    parser.add_argument(
        "--pretrained", type=str, default="stead",
        help="SeisBench 사전학습 가중치 (기본: stead)"
    )

    # 출력
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="출력 파일 경로 (단일 파일 모드)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="출력 디렉토리 경로 (디렉토리 모드)"
    )
    parser.add_argument(
        "--format", "-f", type=str, choices=["json", "csv"],
        default="json", help="출력 포맷 (기본: json)"
    )

    # 추론 옵션
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        choices=["cuda", "cpu", "mps"],
        help="연산 장치 (기본: 자동 선택)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.3,
        help="최소 pick 확률 임계값 (기본: 0.3)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1,
        help="배치 크기 (기본: 1)"
    )

    # DAS 옵션
    parser.add_argument(
        "--channel", type=int, default=0,
        help="DAS 채널 인덱스 (TDMS 파일 전용, 기본: 0)"
    )

    # 시각화
    parser.add_argument(
        "--plot", action="store_true",
        help="결과 시각화 (matplotlib)"
    )

    return parser.parse_args()


def plot_results(waveform, prob_curves, picks, metadata, sampling_rate=100.0):
    """파형과 확률 곡선, pick 결과를 시각화."""
    import matplotlib.pyplot as plt

    n_samples = waveform.shape[1]
    time = [i / sampling_rate for i in range(n_samples)]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # 3성분 파형
    labels = ["Z", "N", "E"]
    for i in range(3):
        axes[0].plot(time, waveform[i], linewidth=0.5, alpha=0.8, label=labels[i])
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"Station: {metadata.get('station', 'N/A')} | "
        f"Start: {metadata.get('start_time', 'N/A')}"
    )

    # P파 확률
    p_curve = prob_curves[1, :n_samples]
    axes[1].plot(time, p_curve, color="blue", linewidth=1.0)
    axes[1].set_ylabel("P probability")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(y=0.3, color="gray", linestyle="--", alpha=0.5)

    # S파 확률
    s_curve = prob_curves[2, :n_samples]
    axes[2].plot(time, s_curve, color="red", linewidth=1.0)
    axes[2].set_ylabel("S probability")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].axhline(y=0.3, color="gray", linestyle="--", alpha=0.5)

    # Noise 확률
    n_curve = prob_curves[0, :n_samples]
    axes[3].plot(time, n_curve, color="green", linewidth=1.0)
    axes[3].set_ylabel("Noise probability")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel("Time (s)")

    # pick 마커 표시
    for pick in picks:
        t = pick["sample_index"] / sampling_rate
        color = "blue" if pick["phase"] == "P" else "red"
        for ax in axes:
            ax.axvline(x=t, color=color, linestyle="--", alpha=0.7, linewidth=1.0)

    plt.tight_layout()
    plt.show()


def plot_results_das(waveform, prob_curves, picks, metadata, sampling_rate=100.0):
    """DAS 파형과 확률 곡선, pick 결과를 시각화."""
    import matplotlib.pyplot as plt

    n_samples = waveform.shape[1]
    time = [i / sampling_rate for i in range(n_samples)]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # DAS 단일 채널 파형
    axes[0].plot(time, waveform[0], linewidth=0.5, alpha=0.8, color="black")
    axes[0].set_ylabel("Amplitude")
    ch_idx = metadata.get("channel_index", "?")
    source = Path(metadata.get("source_file", "")).name
    axes[0].set_title(f"DAS Ch.{ch_idx} | File: {source}")

    # P파 확률
    p_curve = prob_curves[1, :n_samples]
    axes[1].plot(time, p_curve, color="blue", linewidth=1.0)
    axes[1].set_ylabel("P probability")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(y=0.3, color="gray", linestyle="--", alpha=0.5)

    # S파 확률
    s_curve = prob_curves[2, :n_samples]
    axes[2].plot(time, s_curve, color="red", linewidth=1.0)
    axes[2].set_ylabel("S probability")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].axhline(y=0.3, color="gray", linestyle="--", alpha=0.5)

    # Noise 확률
    n_curve = prob_curves[0, :n_samples]
    axes[3].plot(time, n_curve, color="green", linewidth=1.0)
    axes[3].set_ylabel("Noise probability")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel("Time (s)")

    # pick 마커
    for pick in picks:
        t = pick["sample_index"] / sampling_rate
        color = "blue" if pick["phase"] == "P" else "red"
        for ax in axes:
            ax.axvline(x=t, color=color, linestyle="--", alpha=0.7, linewidth=1.0)

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    # SeisBench 모드 vs TPhaseNet 모드
    use_seisbench = args.use_seisbench is not None

    if use_seisbench:
        from inference.seisbench_picker import SeisBenchPicker
        picker = SeisBenchPicker(
            model_name=args.use_seisbench,
            pretrained=args.pretrained,
            device=args.device or "cpu",
            p_threshold=args.threshold,
            s_threshold=args.threshold,
        )
        print(f"Using SeisBench {args.use_seisbench} "
              f"(pretrained: {args.pretrained})")
    else:
        if args.model is None:
            print("Error: --model 또는 --use-seisbench 중 하나를 지정해주세요.")
            sys.exit(1)
        picker = SeismicPicker(
            model_path=args.model,
            config_path=args.config,
            device=args.device,
        )
        if args.threshold != 0.3:
            picker.peak_cfg["min_height"] = args.threshold

    if args.input:
        # TDMS vs mseed 판별
        is_tdms = args.input.lower().endswith(".tdms")

        if is_tdms and use_seisbench:
            print("Error: TDMS 파일은 TPhaseNet 모델만 지원합니다.")
            sys.exit(1)

        # 단일 파일 모드
        print(f"Processing: {args.input}")

        if is_tdms:
            # DAS TDMS 모드
            if args.plot:
                prob_curves, waveform, metadata = picker.get_probabilities_tdms(
                    args.input, channel_index=args.channel
                )
                from inference.postprocessing import extract_picks
                picks = extract_picks(
                    prob_curves,
                    sampling_rate=picker.sampling_rate,
                    min_height=picker.peak_cfg.get("min_height", 0.3),
                    min_distance=picker.peak_cfg.get("min_distance", 100),
                    min_prominence=picker.peak_cfg.get("min_prominence", 0.1),
                )
                plot_results_das(waveform, prob_curves, picks, metadata,
                                 picker.sampling_rate)
                result = {"picks": picks, **metadata}
            else:
                result = picker.pick_tdms(args.input, channel_index=args.channel)

        elif args.plot and not use_seisbench:
            prob_curves, waveform, metadata = picker.get_probabilities(args.input)
            from inference.postprocessing import extract_picks
            from inference.output_formatter import format_picks_absolute

            picks = extract_picks(
                prob_curves,
                sampling_rate=picker.sampling_rate,
                min_height=picker.peak_cfg.get("min_height", 0.3),
                min_distance=picker.peak_cfg.get("min_distance", 100),
                min_prominence=picker.peak_cfg.get("min_prominence", 0.1),
            )
            picks_abs = format_picks_absolute(picks, metadata["start_time"])
            plot_results(waveform, prob_curves, picks, metadata,
                         picker.sampling_rate)
            result = {"picks": picks_abs, **metadata}
        else:
            result = picker.pick(args.input)

        # 출력
        if args.output:
            from inference.output_formatter import to_json as save_json
            save_json(result.get("picks", []), {
                "station": result.get("station", ""),
                "network": result.get("network", ""),
                "location": result.get("location", ""),
                "start_time": result.get("start_time", ""),
                "channels": result.get("channels", []),
            }, args.output)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        # 디렉토리 모드
        mseed_files = sorted(
            glob(os.path.join(args.input_dir, "*.mseed"))
            + glob(os.path.join(args.input_dir, "*.MSEED"))
            + glob(os.path.join(args.input_dir, "*.miniseed"))
        )

        if not mseed_files:
            print(f"No mseed files found in: {args.input_dir}")
            sys.exit(1)

        print(f"Found {len(mseed_files)} mseed files")

        output_dir = args.output_dir or os.path.join(args.input_dir, "picks")
        os.makedirs(output_dir, exist_ok=True)

        results = picker.pick_batch(
            mseed_files,
            output_dir=output_dir,
            output_format=args.format,
        )

        # CSV 출력
        if args.format == "csv":
            csv_data = []
            for r in results:
                if "error" not in r:
                    meta = {
                        "network": r.get("network", ""),
                        "station": r.get("station", ""),
                        "location": r.get("location", ""),
                    }
                    csv_data.append((meta, r.get("picks", [])))

            csv_path = os.path.join(output_dir, "all_picks.csv")
            to_csv(csv_data, csv_path)
            print(f"CSV results saved to: {csv_path}")

        # 요약 출력
        n_success = sum(1 for r in results if "error" not in r)
        n_error = sum(1 for r in results if "error" in r)
        total_picks = sum(
            len(r.get("picks", [])) for r in results if "error" not in r
        )
        print(f"\nSummary: {n_success} files processed, "
              f"{n_error} errors, {total_picks} total picks")


if __name__ == "__main__":
    main()
