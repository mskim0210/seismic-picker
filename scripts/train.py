#!/usr/bin/env python3
"""TPhaseNet 모델 학습 스크립트.

사용법:
    python -m scripts.train \\
        --csv /path/to/merged.csv \\
        --hdf5 /path/to/merged.hdf5 \\
        --config config/default.yaml \\
        --checkpoint-dir ./checkpoints

    # 소량 데이터 테스트
    python -m scripts.train \\
        --csv /path/to/merged.csv \\
        --hdf5 /path/to/merged.hdf5 \\
        --max-samples 1000 \\
        --epochs 5
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.tphasenet import TPhaseNet
from data.stead_dataset import STEADDataset
from data.augmentation import get_default_augmentation
from training.trainer import Trainer
from config.defaults import get_default_config


def parse_args():
    parser = argparse.ArgumentParser(description="TPhaseNet 학습")

    # 데이터
    parser.add_argument("--csv", type=str, required=True,
                        help="STEAD merged.csv 경로")
    parser.add_argument("--hdf5", type=str, required=True,
                        help="STEAD merged.hdf5 경로")

    # 설정
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="설정 YAML 경로")

    # 학습 옵션 (config 오버라이드)
    parser.add_argument("--epochs", type=int, default=None,
                        help="최대 에폭 수")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="배치 크기")
    parser.add_argument("--lr", type=float, default=None,
                        help="학습률")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu", "mps"],
                        help="연산 장치")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 워커 수")

    # 체크포인트
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="체크포인트 저장 디렉토리")
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습할 체크포인트 경로")

    # 디버깅
    parser.add_argument("--max-samples", type=int, default=None,
                        help="최대 샘플 수 제한 (디버깅용)")
    parser.add_argument("--no-augment", action="store_true",
                        help="데이터 증강 비활성화")

    return parser.parse_args()


def main():
    args = parse_args()

    # 설정 로드
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()

    # CLI 인수로 config 오버라이드
    train_cfg = config.setdefault("training", {})
    if args.epochs is not None:
        train_cfg["max_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg.setdefault("optimizer", {})["lr"] = args.lr
    train_cfg["num_workers"] = args.num_workers
    train_cfg["checkpoint_dir"] = args.checkpoint_dir

    # 디바이스 설정
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 데이터 증강
    transform = None if args.no_augment else get_default_augmentation()

    data_cfg = config.get("data", {})
    target_length = data_cfg.get("target_length", 6000)
    sigma = data_cfg.get("label_sigma", 20)

    # 데이터셋 생성
    print("Loading datasets...")
    train_dataset = STEADDataset(
        csv_path=args.csv,
        hdf5_path=args.hdf5,
        split="train",
        target_length=target_length,
        sigma=sigma,
        transform=transform,
        max_samples=args.max_samples,
    )
    val_dataset = STEADDataset(
        csv_path=args.csv,
        hdf5_path=args.hdf5,
        split="val",
        target_length=target_length,
        sigma=sigma,
        transform=None,  # 검증에는 증강 없음
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # 모델 생성
    model = TPhaseNet.from_config(config)
    print(f"Model parameters: {model.count_parameters():,}")

    # 이어서 학습
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # 학습
    trainer = Trainer(model, train_dataset, val_dataset, config, device)

    if args.resume:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    trainer.train()


if __name__ == "__main__":
    main()
