import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_fused import FusedDatasetH5
from model_fused import FusedPhysicsRegressor, fused_loss_with_confidence
# 예시: 필요시 WandB 로깅
# import wandb

class FusedTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # ✅ 1. DataLoader 설정 (기존과 동일)
        self.train_dataset = FusedDatasetH5(config["h5_path"], mode="train", val_ratio=config["val_ratio"])
        self.val_dataset = FusedDatasetH5(config["h5_path"], mode="val", val_ratio=config["val_ratio"])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )

        # ✅ 2. Model 초기화 (기존과 동일)
        sample = self.train_dataset[0]
        q_len = sample["reflectivity"].shape[0]
        n_layers = sample["targets"].shape[0]
        q_values = sample["q_values"]

        self.model = FusedPhysicsRegressor(
            q_len=q_len,
            n_layers=n_layers,
            q_values=q_values,
            physics_weight=config.get("physics_weight", 0.5),
            # fail_safe_mode, physics_dropout은 더 이상 사용 안 함
        ).to(self.device)

        # ✅ 3. Optimizer & Scheduler 개선
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999)  # AdamW 기본값
        )

        # OneCycleLR: 초반 학습 안정성 향상
        if config.get("use_one_cycle", True):
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config["lr"],
                epochs=config["epochs"],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,  # 30% 동안 warmup
                div_factor=25.0,
                final_div_factor=1e4
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=5,
                factor=0.5,
                min_lr=1e-6
            )

        # ✅ 4. Mixed Precision (메모리 절약 & 속도 향상)
        self.scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", True) else None

        # ✅ 5. Checkpoint 관리
        self.ckpt_dir = Path(config["checkpoint_dir"]) / config["exp_name"]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ✅ 6. 메트릭 추적용 버퍼
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # 예시: WandB 초기화
        # if config.get("use_wandb", False):
        #     wandb.init(project="xrr-fused", name=config["exp_name"], config=config)

    def train_epoch(self):
        """학습 1 epoch 실행"""
        self.model.train()
        epoch_loss = 0.0
        physics_confidences = []  # confidence 수집
        physics_contributions = []  # physics_weight * confidence

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            refl = batch["reflectivity"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            # ✅ 7. Mixed Precision Forward
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(refl)
                loss = fused_loss_with_confidence(outputs, targets, self.config["physics_weight"])

            # ✅ 8. Backward & Optimizer Step
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # OneCycleLR은 매 step마다 업데이트
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # ✅ 9. 메트릭 수집
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # confidence 및 실제 기여도 추적
            conf = outputs["confidence"].detach()
            physics_confidences.append(conf.mean().item())
            physics_contributions.append((conf * self.config["physics_weight"]).mean().item())

            # 진행률 표시줄 업데이트
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "conf": f"{conf.mean().item():.3f}"
            })

        # ✅ 10. Epoch 통계 계산
        avg_loss = epoch_loss / len(self.train_loader)
        avg_confidence = np.mean(physics_confidences)
        avg_physics_contrib = np.mean(physics_contributions)

        return avg_loss, {
            "confidence": avg_confidence,
            "physics_contrib": avg_physics_contrib
        }

    def validate(self):
        """검증 실행"""
        self.model.eval()
        epoch_loss = 0.0
        physics_confidences = []
        param_errors = defaultdict(list)  # 두께, 거칠기, SLD별 오차

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                refl = batch["reflectivity"].to(self.device, non_blocking=True)
                targets = batch["targets"].to(self.device, non_blocking=True)

                # 검증시에는 항상 full precision
                outputs = self.model(refl)
                loss = fused_loss_with_confidence(outputs, targets, self.config["physics_weight"])

                epoch_loss += loss.item()

                # 물리 모델 신뢰도 수집
                conf = outputs["confidence"].detach()
                physics_confidences.append(conf.mean().item())

                # 파라미터별 오차 계산 (역정규화된 값 기준)
                final_pred = outputs["final"]
                mae = torch.abs(final_pred - targets).mean(dim=0)  # (n_layers, 3)

                param_errors["thickness"].append(mae[:, 0].mean().item())
                param_errors["roughness"].append(mae[:, 1].mean().item())
                param_errors["sld"].append(mae[:, 2].mean().item())

        # ✅ 11. 검증 통계
        avg_loss = epoch_loss / len(self.val_loader)
        avg_confidence = np.mean(physics_confidences)

        # 파라미터별 평균 오차
        param_mae = {k: np.mean(v) for k, v in param_errors.items()}

        return avg_loss, {
            "confidence": avg_confidence,
            "param_mae": param_mae
        }

    def train(self):
        """전체 학습 루프"""
        best_val_loss = float("inf")
        patience = self.config.get("patience", 10)
        no_improve_epochs = 0

        print(f"🚀 Training started on {self.device}")
        print(f"📊 Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

        for epoch in range(1, self.config["epochs"] + 1):
            # ✅ 12. Train & Validate
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            # ✅ 13. ReduceLROnPlateau 업데이트 (OneCycle이 아닐 때)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            # ✅ 14. 상세 로깅
            print(
                f"[Epoch {epoch:02d}/{self.config['epochs']}] "
                f"Loss: {train_loss:.4f} → {val_loss:.4f} | "
                f"Conf: {train_metrics['confidence']:.3f} → {val_metrics['confidence']:.3f} | "
                f"Physics Contrib: {train_metrics['physics_contrib']:.3f}"
            )

            # 파라미터별 오차 출력 (검증 시)
            if "param_mae" in val_metrics:
                pmae = val_metrics["param_mae"]
                print(f"  → MAE: Thickness={pmae['thickness']:.3f}, Roughness={pmae['roughness']:.3f}, SLD={pmae['sld']:.3f}")

            # ✅ 15. WandB 로깅 (선택)
            # if self.config.get("use_wandb", False):
            #     wandb.log({
            #         "epoch": epoch,
            #         "train/loss": train_loss,
            #         "val/loss": val_loss,
            #         "train/confidence": train_metrics["confidence"],
            #         "val/confidence": val_metrics["confidence"],
            #         **{f"val/mae_{k}": v for k, v in val_metrics.get("param_mae", {}).items()}
            #     })

            # ✅ 16. Checkpoint 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                self.save_checkpoint("best.pt", epoch, val_loss, val_metrics)
                print("  💾 Best model saved!")
            else:
                no_improve_epochs += 1

            # 주기적 저장
            if epoch % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt", epoch, val_loss, val_metrics)

            # ✅ 17. Early Stopping
            if no_improve_epochs >= patience:
                print(f"⏹️ Early stopping triggered after {patience} epochs without improvement")
                break

        print(f"✅ Training finished. Best val loss: {best_val_loss:.4f}")

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, metrics: dict):
        """체크포인트 저장"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "val_loss": val_loss,
            "config": self.config,
            "metrics": metrics
        }

        torch.save(checkpoint, self.ckpt_dir / filename)
        print(f"  💾 Saved: {self.ckpt_dir / filename}")

    def load_checkpoint(self, filepath: str):
        """체크포인트 로드 (추가)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"✅ Loaded checkpoint from {filepath}")
