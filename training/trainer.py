import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_pick_metrics


class Trainer:
    """TPhaseNet 학습 관리자.

    Args:
        model: TPhaseNet 모델
        train_dataset: 학습 Dataset
        val_dataset: 검증 Dataset
        config: 학습 설정 dict
        device: torch.device
    """

    def __init__(self, model, train_dataset, val_dataset, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        train_cfg = config.get("training", {})

        # DataLoader
        from ..data.stead_dataset import worker_init_fn
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("batch_size", 64),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 64),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        # 손실함수
        loss_type = train_cfg.get("loss", "weighted_ce")
        if loss_type == "weighted_ce":
            from .losses import WeightedCrossEntropyLoss
            class_weights = train_cfg.get("class_weights", [1.0, 30.0, 30.0])
            self.criterion = WeightedCrossEntropyLoss(class_weights).to(self.device)
        else:
            from .losses import FocalCrossEntropyLoss
            gamma = train_cfg.get("focal_gamma", 2.0)
            self.criterion = FocalCrossEntropyLoss(gamma=gamma).to(self.device)

        # 옵티마이저
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 1e-5),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # 스케줄러
        sched_cfg = train_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "reduce_on_plateau")
        if sched_type == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 5),
                min_lr=sched_cfg.get("min_lr", 1e-6),
            )
        else:
            total_steps = (train_cfg.get("max_epochs", 100)
                           * len(self.train_loader))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=opt_cfg.get("lr", 1e-3),
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy="cos",
            )

        self.sched_type = sched_type
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.early_stop_patience = train_cfg.get("early_stopping", {}).get(
            "patience", 15
        )
        self.use_amp = train_cfg.get("mixed_precision", True)
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if (
            self.use_amp and self.device.type == "cuda"
        ) else None

        # 상태
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self):
        """전체 학습 루프 실행."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Val samples: {len(self.val_loader.dataset):,}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Max epochs: {self.max_epochs}")
        print("-" * 60)

        for epoch in range(1, self.max_epochs + 1):
            start = time.time()

            # 학습
            train_loss = self._train_epoch(epoch)

            # 검증
            val_loss, metrics = self._validate(epoch)

            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"P-F1: {metrics.get('p_f1', 0):.3f} | "
                f"S-F1: {metrics.get('s_f1', 0):.3f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

            # 스케줄러 업데이트
            if self.sched_type == "reduce_on_plateau":
                self.scheduler.step(val_loss)

            # 체크포인트
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, metrics, is_best=True)
                print(f"  -> Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            # 주기적 저장
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_loss, metrics, is_best=False)

            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {self.early_stop_patience} epochs)")
                break

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def _train_epoch(self, epoch):
        """1 에폭 학습."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]",
                     leave=False)

        for waveforms, labels in pbar:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    predictions = self.model(waveforms)
                    loss = self.criterion(predictions, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(waveforms)
                loss = self.criterion(predictions, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.gradient_clip)
                self.optimizer.step()

            if self.sched_type == "one_cycle":
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch):
        """검증 및 pick 메트릭 계산."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        for waveforms, labels in tqdm(self.val_loader,
                                       desc=f"Epoch {epoch} [Val]",
                                       leave=False):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(waveforms)
            loss = self.criterion(predictions, labels)

            total_loss += loss.item()
            n_batches += 1

            # 메트릭 계산을 위해 수집 (첫 10배치만)
            if n_batches <= 10:
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = total_loss / max(n_batches, 1)

        # Pick 메트릭
        metrics = {}
        if all_preds:
            preds = np.concatenate(all_preds, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            metrics = compute_pick_metrics(preds, labels)

        return val_loss, metrics

    def _save_checkpoint(self, epoch, val_loss, metrics, is_best=False):
        """체크포인트 저장."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "metrics": metrics,
            "config": self.config,
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")

        torch.save(checkpoint, path)
