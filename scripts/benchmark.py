#!/usr/bin/env python3
"""모델 간 벤치마크 비교 스크립트.

TPhaseNet과 SeisBench 사전학습 모델(PhaseNet, EQTransformer)을
동일 데이터셋에서 비교 평가.

사용법:
    # TPhaseNet vs SeisBench PhaseNet vs EQTransformer 비교
    python -m scripts.benchmark \
        --model checkpoints/best_model.pt \
        --csv /path/to/merged.csv \
        --hdf5 /path/to/merged.hdf5

    # 특정 모델만 비교 (seisbench 제외)
    python -m scripts.benchmark \
        --model checkpoints/best_model.pt \
        --csv /path/to/merged.csv \
        --hdf5 /path/to/merged.hdf5 \
        --no-seisbench

    # 추론 속도만 측정
    python -m scripts.benchmark \
        --model checkpoints/best_model.pt \
        --csv /path/to/merged.csv \
        --hdf5 /path/to/merged.hdf5 \
        --speed-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.tphasenet import TPhaseNet
from data.stead_dataset import STEADDataset
from training.metrics import compute_pick_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="모델 벤치마크 비교")

    parser.add_argument("--model", "-m", type=str, default=None,
                        help="TPhaseNet 체크포인트 경로 (없으면 SeisBench만 비교)")
    parser.add_argument("--csv", type=str, required=True,
                        help="STEAD merged.csv 경로")
    parser.add_argument("--hdf5", type=str, required=True,
                        help="STEAD merged.hdf5 경로")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="최대 샘플 수 (빠른 비교용)")

    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="허용 오차 (초)")

    parser.add_argument("--no-seisbench", action="store_true",
                        help="SeisBench 모델 비교 생략")
    parser.add_argument("--speed-only", action="store_true",
                        help="추론 속도만 측정 (메트릭 생략)")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="결과 저장 디렉토리")

    return parser.parse_args()


def get_device(device_str):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def benchmark_tphasenet(model_path, dataloader, device, threshold, tolerance_sec,
                        sampling_rate, speed_only=False):
    """TPhaseNet 모델 벤치마크."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", _default_config())

    model = TPhaseNet.from_config(config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    param_count = model.count_parameters()
    tolerance_samples = int(tolerance_sec * sampling_rate)

    all_preds = []
    all_labels = []
    total_time = 0.0
    total_samples = 0

    # Warmup
    with torch.no_grad():
        dummy = torch.randn(1, 3, 6000).to(device)
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="TPhaseNet"):
            waveforms = waveforms.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predictions = model(waveforms)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            total_samples += waveforms.shape[0]

            if not speed_only:
                all_preds.append(predictions.cpu())
                all_labels.append(labels)

    result = {
        "model": "TPhaseNet",
        "parameters": param_count,
        "total_samples": total_samples,
        "total_time_sec": round(total_time, 3),
        "throughput_samples_per_sec": round(total_samples / total_time, 1),
        "avg_latency_ms": round(total_time / total_samples * 1000, 2),
    }

    if not speed_only and all_preds:
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_pick_metrics(preds, labels, threshold=threshold,
                                       tolerance_samples=tolerance_samples,
                                       sampling_rate=sampling_rate)
        result.update(metrics)

    return result


def benchmark_seisbench(model_name, pretrained, dataloader, device_str,
                        threshold, tolerance_sec, sampling_rate,
                        speed_only=False):
    """SeisBench 모델 벤치마크."""
    try:
        import seisbench.models as sbm
    except ImportError:
        print(f"  [SKIP] seisbench not installed, skipping {model_name}")
        return None

    device = torch.device(device_str) if device_str else torch.device("cpu")

    if model_name == "PhaseNet":
        model = sbm.PhaseNet.from_pretrained(pretrained)
    elif model_name == "EQTransformer":
        model = sbm.EQTransformer.from_pretrained(pretrained)
    else:
        print(f"  [SKIP] Unknown model: {model_name}")
        return None

    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tolerance_samples = int(tolerance_sec * sampling_rate)

    all_preds = []
    all_labels = []
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc=f"SeisBench {model_name}"):
            waveforms = waveforms.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predictions = model(waveforms)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            total_samples += waveforms.shape[0]

            if not speed_only:
                # SeisBench 출력 형식을 (B, 3, T)로 맞춤
                if predictions.dim() == 3 and predictions.shape[1] == 3:
                    all_preds.append(predictions.cpu())
                else:
                    all_preds.append(predictions.cpu())
                all_labels.append(labels)

    result = {
        "model": f"SeisBench-{model_name} ({pretrained})",
        "parameters": param_count,
        "total_samples": total_samples,
        "total_time_sec": round(total_time, 3),
        "throughput_samples_per_sec": round(total_samples / total_time, 1),
        "avg_latency_ms": round(total_time / total_samples * 1000, 2),
    }

    if not speed_only and all_preds:
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        # 출력 shape이 다를 수 있으므로 확인
        if preds.shape == labels.shape:
            metrics = compute_pick_metrics(preds, labels, threshold=threshold,
                                           tolerance_samples=tolerance_samples,
                                           sampling_rate=sampling_rate)
            result.update(metrics)
        else:
            result["note"] = (f"Output shape mismatch: "
                              f"pred={list(preds.shape)}, label={list(labels.shape)}")

    return result


def print_comparison(results):
    """벤치마크 결과 비교표 출력."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    # 속도 비교
    print(f"\n{'Model':<35} {'Params':>10} {'Throughput':>14} {'Latency':>12}")
    print("-" * 75)
    for r in results:
        name = r["model"]
        params = f"{r['parameters']:,}"
        throughput = f"{r['throughput_samples_per_sec']:.1f} samp/s"
        latency = f"{r['avg_latency_ms']:.2f} ms"
        print(f"{name:<35} {params:>10} {throughput:>14} {latency:>12}")

    # 정확도 비교 (메트릭이 있는 경우만)
    has_metrics = any("p_f1" in r for r in results)
    if has_metrics:
        print(f"\n{'Model':<35} {'P-Prec':>8} {'P-Rec':>8} {'P-F1':>8}"
              f"  {'S-Prec':>8} {'S-Rec':>8} {'S-F1':>8}")
        print("-" * 90)
        for r in results:
            name = r["model"]
            if "p_f1" in r:
                print(f"{name:<35} "
                      f"{r['p_precision']:>8.4f} {r['p_recall']:>8.4f} "
                      f"{r['p_f1']:>8.4f}  "
                      f"{r['s_precision']:>8.4f} {r['s_recall']:>8.4f} "
                      f"{r['s_f1']:>8.4f}")
            elif "note" in r:
                print(f"{name:<35} {r['note']}")

        print(f"\n{'Model':<35} {'P-Residual (sec)':>20} {'S-Residual (sec)':>20}")
        print("-" * 80)
        for r in results:
            name = r["model"]
            if "p_mean_residual_sec" in r:
                p_res = f"{r['p_mean_residual_sec']:+.4f} ± {r['p_std_residual_sec']:.4f}"
                s_res = f"{r['s_mean_residual_sec']:+.4f} ± {r['s_std_residual_sec']:.4f}"
                print(f"{name:<35} {p_res:>20} {s_res:>20}")

    print("=" * 80)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # 데이터셋
    sampling_rate = 100.0
    print("Loading test dataset...")
    dataset = STEADDataset(
        csv_path=args.csv,
        hdf5_path=args.hdf5,
        split="test",
        target_length=6000,
        sigma=20,
        transform=None,
        max_samples=args.max_samples,
    )
    print(f"  Test samples: {len(dataset):,}")

    from data.stead_dataset import worker_init_fn
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    results = []

    # 1) TPhaseNet
    if args.model:
        print("\n--- TPhaseNet ---")
        r = benchmark_tphasenet(
            args.model, dataloader, device,
            threshold=args.threshold,
            tolerance_sec=args.tolerance,
            sampling_rate=sampling_rate,
            speed_only=args.speed_only,
        )
        results.append(r)

    # 2) SeisBench 모델들
    if not args.no_seisbench:
        device_str = str(device)
        for model_name in ["PhaseNet", "EQTransformer"]:
            print(f"\n--- SeisBench {model_name} ---")
            r = benchmark_seisbench(
                model_name, "stead", dataloader, device_str,
                threshold=args.threshold,
                tolerance_sec=args.tolerance,
                sampling_rate=sampling_rate,
                speed_only=args.speed_only,
            )
            if r is not None:
                results.append(r)

    # 결과 출력
    if results:
        print_comparison(results)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, "benchmark_results.json")
            with open(path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {path}")
    else:
        print("No models to benchmark. Use --model or remove --no-seisbench.")


def _default_config():
    return {
        "model": {
            "name": "TPhaseNet",
            "in_channels": 3,
            "classes": 3,
            "filters_root": 8,
            "depth": 5,
            "kernel_size": 7,
            "stride": 2,
            "transformer_start_level": 3,
            "n_heads": 4,
            "ff_dim_factor": 4,
            "dropout": 0.1,
        },
        "data": {
            "target_length": 6000,
            "sampling_rate": 100.0,
            "label_sigma": 20,
        },
    }


if __name__ == "__main__":
    main()
