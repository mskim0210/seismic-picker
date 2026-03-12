#!/usr/bin/env python3
"""DAS 전체 채널 추론 및 2D 섹션 시각화.

전체 DAS 채널에 대해 TPhaseNet 추론을 수행하고,
채널 vs 시간 2D 섹션으로 P/S 확률을 시각화.
"""

import sys
import numpy as np
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.tdms_loader import load_tdms_channel
from inference.picker import SeismicPicker


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DAS 전체 채널 추론 + 2D 시각화")
    parser.add_argument("--input", "-i", required=True, help="TDMS 파일 경로")
    parser.add_argument("--model", "-m", required=True, help="모델 체크포인트")
    parser.add_argument("--device", "-d", default=None)
    parser.add_argument("--threshold", "-t", type=float, default=0.3)
    parser.add_argument("--save", "-s", default=None, help="저장 경로 (png)")
    args = parser.parse_args()

    # 모델 로드
    picker = SeismicPicker(model_path=args.model, device=args.device)
    if args.threshold != 0.3:
        picker.peak_cfg["min_height"] = args.threshold

    # TDMS 메타데이터 확인
    _, meta0 = load_tdms_channel(args.input, channel_index=0,
                                  target_sampling_rate=picker.sampling_rate)
    n_channels = meta0["n_channels_total"]
    n_samples = meta0["n_samples"]
    sr = meta0["sampling_rate"]

    print(f"TDMS: {n_channels} channels, {n_samples} samples @ {sr}Hz")
    print(f"Processing all {n_channels} channels...")

    # 전체 채널 추론
    waveform_section = np.zeros((n_channels, n_samples), dtype=np.float32)
    p_prob_section = np.zeros((n_channels, n_samples), dtype=np.float32)
    s_prob_section = np.zeros((n_channels, n_samples), dtype=np.float32)
    p_picks_list = []
    s_picks_list = []

    data_cfg = picker.config.get("data", {})

    for ch in range(n_channels):
        if ch % 100 == 0:
            print(f"  Channel {ch}/{n_channels}...")

        waveform, _ = load_tdms_channel(
            args.input, channel_index=ch,
            target_sampling_rate=sr, config=data_cfg,
        )

        # 추론
        prob_curves = picker._infer_single(waveform, picker.target_length)
        prob_curves = prob_curves[:, :n_samples]

        waveform_section[ch] = waveform[0]  # 단일성분
        p_prob_section[ch] = prob_curves[1]
        s_prob_section[ch] = prob_curves[2]

        # pick 추출
        from inference.postprocessing import extract_picks
        picks = extract_picks(
            prob_curves, sampling_rate=sr,
            min_height=args.threshold,
            min_distance=int(sr),
            min_prominence=0.1,
        )
        for p in picks:
            if p["phase"] == "P":
                p_picks_list.append((ch, p["time_offset_sec"], p["confidence"]))
            else:
                s_picks_list.append((ch, p["time_offset_sec"], p["confidence"]))

    print(f"Done. P picks: {len(p_picks_list)}, S picks: {len(s_picks_list)}")

    # 시각화
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = np.arange(n_samples) / sr
    ch_axis = np.arange(n_channels)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

    # 1) DAS 파형 섹션
    vmax = np.percentile(np.abs(waveform_section), 99)
    axes[0].imshow(waveform_section, aspect="auto", cmap="seismic",
                   vmin=-vmax, vmax=vmax,
                   extent=[time_axis[0], time_axis[-1], n_channels-1, 0])
    axes[0].set_title("DAS Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Channel")

    # 2) P파 확률 섹션
    axes[1].imshow(p_prob_section, aspect="auto", cmap="Blues",
                   vmin=0, vmax=1,
                   extent=[time_axis[0], time_axis[-1], n_channels-1, 0])
    if p_picks_list:
        p_ch, p_t, p_c = zip(*p_picks_list)
        axes[1].scatter(p_t, p_ch, c="blue", s=1, alpha=0.5)
    axes[1].set_title(f"P Probability (picks: {len(p_picks_list)})")
    axes[1].set_xlabel("Time (s)")

    # 3) S파 확률 섹션
    im = axes[2].imshow(s_prob_section, aspect="auto", cmap="Reds",
                        vmin=0, vmax=1,
                        extent=[time_axis[0], time_axis[-1], n_channels-1, 0])
    if s_picks_list:
        s_ch, s_t, s_c = zip(*s_picks_list)
        axes[2].scatter(s_t, s_ch, c="red", s=1, alpha=0.5)
    axes[2].set_title(f"S Probability (picks: {len(s_picks_list)})")
    axes[2].set_xlabel("Time (s)")

    fig.suptitle(f"DAS Section: {Path(args.input).name}", fontsize=14)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
