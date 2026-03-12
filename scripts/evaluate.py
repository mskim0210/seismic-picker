#!/usr/bin/env python3
"""TPhaseNet 모델 평가 스크립트.

학습된 모델을 STEAD 테스트셋에서 평가하여 P/S파 검출 성능을 정량적으로 측정.

사용법:
    # 기본 평가
    python -m scripts.evaluate \
        --model checkpoints/best_model.pt \
        --csv /path/to/merged.csv \
        --hdf5 /path/to/merged.hdf5

    # threshold 범위 분석
    python -m scripts.evaluate \
        --model checkpoints/best_model.pt \
        --csv /path/to/merged.csv \
        --hdf5 /path/to/merged.hdf5 \
        --sweep-thresholds

    # 결과 저장
    python -m scripts.evaluate \
        --model checkpoints/best_model.pt \
        --csv /path/to/merged.csv \
        --hdf5 /path/to/merged.hdf5 \
        --output-dir ./eval_results
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
from config.defaults import get_default_config


def parse_args():
    parser = argparse.ArgumentParser(description="TPhaseNet 모델 평가")

    # 필수
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="학습된 모델 체크포인트 경로 (.pt)")
    parser.add_argument("--csv", type=str, required=True,
                        help="STEAD merged.csv 경로")
    parser.add_argument("--hdf5", type=str, required=True,
                        help="STEAD merged.hdf5 경로")

    # 옵션
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="설정 YAML 경로 (None이면 체크포인트 내장 설정 사용)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"],
                        help="평가할 데이터 분할 (기본: test)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="배치 크기 (기본: 64)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu", "mps"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="최대 샘플 수 제한 (빠른 테스트용)")

    # 평가 파라미터
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="pick 확률 임계값 (기본: 0.3)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="허용 오차 (초, 기본: 0.5)")

    # threshold sweep
    parser.add_argument("--sweep-thresholds", action="store_true",
                        help="여러 threshold에서 성능 변화 분석")

    # 출력
    parser.add_argument("--output-dir", type=str, default=None,
                        help="결과 저장 디렉토리")

    return parser.parse_args()


def load_model(model_path, config, device):
    """체크포인트에서 모델 로드."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 체크포인트에 config가 있으면 사용
    if config is None and "config" in checkpoint:
        config = checkpoint["config"]
    if config is None:
        config = get_default_config()

    model = TPhaseNet.from_config(config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", checkpoint.get("best_val_loss", "?"))
    print(f"Loaded model from epoch {epoch} (val_loss: {val_loss})")

    return model, config


def evaluate(model, dataloader, device, threshold=0.3, tolerance_sec=0.5,
             sampling_rate=100.0):
    """전체 데이터셋에서 모델 평가."""
    tolerance_samples = int(tolerance_sec * sampling_rate)

    all_preds = []
    all_labels = []
    total_loss_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Evaluating"):
            waveforms = waveforms.to(device)

            start_t = time.perf_counter()
            predictions = model(waveforms)
            elapsed = time.perf_counter() - start_t

            total_loss_time += elapsed
            total_samples += waveforms.shape[0]

            all_preds.append(predictions.cpu())
            all_labels.append(labels)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # pick 메트릭 계산
    metrics = compute_pick_metrics(
        preds, labels,
        threshold=threshold,
        tolerance_samples=tolerance_samples,
        sampling_rate=sampling_rate,
    )

    # 추론 속도
    throughput = total_samples / total_loss_time if total_loss_time > 0 else 0
    metrics["throughput_samples_per_sec"] = round(throughput, 1)
    metrics["total_samples"] = total_samples
    metrics["threshold"] = threshold
    metrics["tolerance_sec"] = tolerance_sec

    return metrics


def sweep_thresholds(model, dataloader, device, tolerance_sec=0.5,
                     sampling_rate=100.0):
    """여러 threshold에서 성능 측정."""
    tolerance_samples = int(tolerance_sec * sampling_rate)

    # 먼저 전체 예측을 수집
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Collecting predictions"):
            waveforms = waveforms.to(device)
            predictions = model(waveforms)
            all_preds.append(predictions.cpu())
            all_labels.append(labels)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # 여러 threshold에서 메트릭 계산
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    for thr in thresholds:
        metrics = compute_pick_metrics(
            preds, labels,
            threshold=thr,
            tolerance_samples=tolerance_samples,
            sampling_rate=sampling_rate,
        )
        metrics["threshold"] = thr
        results.append(metrics)
        print(f"  threshold={thr:.2f} | "
              f"P-F1={metrics['p_f1']:.3f} | S-F1={metrics['s_f1']:.3f} | "
              f"P-Prec={metrics['p_precision']:.3f} | P-Rec={metrics['p_recall']:.3f} | "
              f"S-Prec={metrics['s_precision']:.3f} | S-Rec={metrics['s_recall']:.3f}")

    return results


def print_results(metrics):
    """평가 결과 출력."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  Threshold: {metrics['threshold']}")
    print(f"  Tolerance: {metrics['tolerance_sec']}s "
          f"({int(metrics['tolerance_sec'] * 100)} samples)")
    print(f"  Total samples: {metrics['total_samples']:,}")
    print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")

    print(f"\n  {'Phase':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}"
          f"    {'TP':>6} {'FP':>6} {'FN':>6}")
    print("  " + "-" * 58)

    for phase in ["p", "s"]:
        name = phase.upper()
        prec = metrics[f"{phase}_precision"]
        rec = metrics[f"{phase}_recall"]
        f1 = metrics[f"{phase}_f1"]
        tp = metrics[f"{phase}_tp"]
        fp = metrics[f"{phase}_fp"]
        fn = metrics[f"{phase}_fn"]
        print(f"  {name:<8} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}"
              f"    {tp:>6} {fp:>6} {fn:>6}")

    print(f"\n  Arrival Time Residuals:")
    for phase in ["p", "s"]:
        name = phase.upper()
        mean = metrics[f"{phase}_mean_residual_sec"]
        std = metrics[f"{phase}_std_residual_sec"]
        print(f"    {name}-wave: {mean:+.4f} ± {std:.4f} sec")

    print("=" * 60)


def main():
    args = parse_args()

    # 디바이스
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # config 로드
    config = None
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # 모델 로드
    model, config = load_model(args.model, config, device)

    # 데이터셋
    data_cfg = config.get("data", {})
    target_length = data_cfg.get("target_length", 6000)
    sigma = data_cfg.get("label_sigma", 20)
    sampling_rate = data_cfg.get("sampling_rate", 100.0)

    print(f"Loading {args.split} dataset...")
    dataset = STEADDataset(
        csv_path=args.csv,
        hdf5_path=args.hdf5,
        split=args.split,
        target_length=target_length,
        sigma=sigma,
        transform=None,
        max_samples=args.max_samples,
    )
    print(f"  {args.split} samples: {len(dataset):,}")

    from data.stead_dataset import worker_init_fn
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # 평가
    if args.sweep_thresholds:
        print("\nThreshold sweep:")
        sweep_results = sweep_thresholds(
            model, dataloader, device,
            tolerance_sec=args.tolerance,
            sampling_rate=sampling_rate,
        )

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, "threshold_sweep.json")
            with open(path, "w") as f:
                json.dump(sweep_results, f, indent=2)
            print(f"\nSweep results saved to: {path}")
    else:
        metrics = evaluate(
            model, dataloader, device,
            threshold=args.threshold,
            tolerance_sec=args.tolerance,
            sampling_rate=sampling_rate,
        )
        print_results(metrics)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
