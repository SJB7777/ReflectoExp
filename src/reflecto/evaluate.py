import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from reflecto.dataset import DatasetH5, ParamQuantizer
from reflecto.model import XRRClassifier


class XRREvaluator:
    def __init__(
        self,
        h5_path: str,
        checkpoint_path: str,
        batch_size: int = 64,
        device: str | torch.device = None,
        num_workers: int = 0,   # Windows에서는 0~2 권장
        pin_memory: bool = True,
    ):
        self.h5_path = h5_path
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Quantizer
        self.quantizer = ParamQuantizer(thickness_bins=None, roughness_bins=None, sld_bins=None)
        self.n_th = len(self.quantizer.thickness_bins) - 1
        self.n_rg = len(self.quantizer.roughness_bins) - 1
        self.n_sld = len(self.quantizer.sld_bins) - 1

        # Dataset
        self.val_dataset = DatasetH5(self.h5_path, self.quantizer, device="cpu")
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        # Model
        sample_refl, sample_label = self.val_dataset[0]
        self.q_len = sample_refl.shape[0]
        self.n_layers = sample_label.shape[0]

        self.model = XRRClassifier(
            q_len=self.q_len,
            n_layers=self.n_layers,
            n_th_bins=self.n_th,
            n_rg_bins=self.n_rg,
            n_sld_bins=self.n_sld
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.eval()

        # Loss
        self.ce = nn.CrossEntropyLoss()

    def evaluate(self, use_amp: bool = False, logits_clip: float = 50.0):
        """모든 평가 데이터를 계산하고 결과 저장"""
        all_tgt_th, all_tgt_rg, all_tgt_sld = [], [], []
        all_pred_th, all_pred_rg, all_pred_sld = [], [], []
        all_losses = []

        self.model.eval()

        pbar = tqdm(self.val_loader, desc="Evaluating")
        with torch.no_grad():
            for refl, labels in pbar:
                refl = refl.to(self.device)
                labels = labels.to(self.device)

                # --- AMP 실제 적용 ---
                if use_amp and self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        th_logits, rg_logits, sld_logits = self.model(refl)
                else:
                    th_logits, rg_logits, sld_logits = self.model(refl)

                # --- logits clipping (학습과 동일하게) ---
                if logits_clip is not None:
                    th_logits = th_logits.clamp(-logits_clip, logits_clip)
                    rg_logits = rg_logits.clamp(-logits_clip, logits_clip)
                    sld_logits = sld_logits.clamp(-logits_clip, logits_clip)

                # --- target ---
                tgt_th = labels[:, :, 0]
                tgt_rg = labels[:, :, 1]
                tgt_sld = labels[:, :, 2]

                # --- Loss 계산(CrossEntropy 조합) ---
                loss = (
                    self.ce(th_logits.reshape(-1, th_logits.shape[-1]),
                            tgt_th.reshape(-1)) +
                    self.ce(rg_logits.reshape(-1, rg_logits.shape[-1]),
                            tgt_rg.reshape(-1)) +
                    self.ce(sld_logits.reshape(-1, sld_logits.shape[-1]),
                            tgt_sld.reshape(-1))
                )
                all_losses.append(loss.item())

                # --- 예측 저장 ---
                all_tgt_th.append(tgt_th.cpu().numpy())
                all_tgt_rg.append(tgt_rg.cpu().numpy())
                all_tgt_sld.append(tgt_sld.cpu().numpy())

                all_pred_th.append(th_logits.argmax(dim=-1).cpu().numpy())
                all_pred_rg.append(rg_logits.argmax(dim=-1).cpu().numpy())
                all_pred_sld.append(sld_logits.argmax(dim=-1).cpu().numpy())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- Flatten ---
        self.all_tgt_th = np.concatenate(all_tgt_th).ravel()
        self.all_tgt_rg = np.concatenate(all_tgt_rg).ravel()
        self.all_tgt_sld = np.concatenate(all_tgt_sld).ravel()

        self.all_pred_th = np.concatenate(all_pred_th).ravel()
        self.all_pred_rg = np.concatenate(all_pred_rg).ravel()
        self.all_pred_sld = np.concatenate(all_pred_sld).ravel()

        self.all_losses = all_losses

        # --- Accuracy ---
        self.acc_th = (self.all_pred_th == self.all_tgt_th).mean()
        self.acc_rg = (self.all_pred_rg == self.all_tgt_rg).mean()
        self.acc_sld = (self.all_pred_sld == self.all_tgt_sld).mean()
        self.avg_loss = np.mean(all_losses)

        return {
            "loss": self.avg_loss,
            "acc_th": self.acc_th,
            "acc_rg": self.acc_rg,
            "acc_sld": self.acc_sld
        }

    def plot_loss(self, window: int = 50):
        losses = np.array(self.all_losses)
        plt.figure(figsize=(8, 4))

        # Raw loss
        plt.plot(losses, alpha=0.4, label="Raw Loss")

        # Moving average
        if len(losses) >= window:
            ma = np.convolve(losses, np.ones(window)/window, mode="valid")
            plt.plot(range(window-1, window-1+len(ma)), ma, linewidth=2, label=f"MA({window})")

        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Validation Loss (Raw + Moving Average)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_error_histogram(self):
        plt.figure(figsize=(15, 4))

        for i, (tgt, pred, name) in enumerate(zip(
            [self.all_tgt_th, self.all_tgt_rg, self.all_tgt_sld],
            [self.all_pred_th, self.all_pred_rg, self.all_pred_sld],
            ["Thickness", "Roughness", "SLD"], strict=False
        )):
            plt.subplot(1, 3, i+1)
            errors = pred - tgt

            sns.histplot(errors, bins=40, kde=True)
            mean = errors.mean()
            std = errors.std()

            plt.axvline(mean, color="r", linestyle="--", label=f"Mean = {mean:.2f}")
            plt.axvline(mean+std, color="g", linestyle=":", label=f"+1σ = {mean+std:.2f}")
            plt.axvline(mean-std, color="g", linestyle=":", label=f"-1σ = {mean-std:.2f}")

            plt.title(f"{name} Error Distribution")
            plt.xlabel("Prediction - Target")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_scatter(self, use_hexbin: bool = False):
        plt.figure(figsize=(15, 4))

        for i, (tgt, pred, name) in enumerate(zip(
            [self.all_tgt_th, self.all_tgt_rg, self.all_tgt_sld],
            [self.all_pred_th, self.all_pred_rg, self.all_pred_sld],
            ["Thickness", "Roughness", "SLD"], strict=False
        )):
            plt.subplot(1, 3, i+1)

            if use_hexbin:
                plt.hexbin(tgt, pred, gridsize=40, cmap="viridis", mincnt=1)
                plt.colorbar(label="Density")
            else:
                plt.scatter(tgt, pred, alpha=0.2, s=8)

            # y = x선 (이상적 예측)
            tmin, tmax = tgt.min(), tgt.max()
            plt.plot([tmin, tmax], [tmin, tmax], "r--", linewidth=1.5, label="Ideal")

            # 추세선 (회귀)
            coef = np.polyfit(tgt, pred, 1)
            poly = np.poly1d(coef)
            plt.plot([tmin, tmax], poly([tmin, tmax]), "b-", linewidth=1.2, label=f"Fit: y={coef[0]:.2f}x+{coef[1]:.2f}")

            plt.title(f"{name}: Target vs Predicted")
            plt.xlabel("Target")
            plt.ylabel("Predicted")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_correlation(self):
        import pandas as pd

        df_target = pd.DataFrame({
            "Thickness": self.all_tgt_th,
            "Roughness": self.all_tgt_rg,
            "SLD": self.all_tgt_sld
        })

        df_pred = pd.DataFrame({
            "Thickness": self.all_pred_th,
            "Roughness": self.all_pred_rg,
            "SLD": self.all_pred_sld
        })

        df_error = df_pred - df_target

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(df_target.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title("Target Correlation")

        sns.heatmap(df_pred.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title("Prediction Correlation")

        sns.heatmap(df_error.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
        axes[2].set_title("Prediction Error Correlation")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    evaluator = XRREvaluator(
        h5_path=r"D:\03_Resources\Data\XRR_AI\data\p300o6_raw.h5",
        checkpoint_path="xrr_model.pt",
        batch_size=64,
        num_workers=0  # Windows 안정성
    )
    metrics = evaluator.evaluate(use_amp=True)
    print(f"Validation Loss: {metrics['loss']:.4f}")
    print(f"Accuracy - Thickness: {metrics['acc_th']:.3f}, Roughness: {metrics['acc_rg']:.3f}, SLD: {metrics['acc_sld']:.3f}")

    evaluator.plot_loss()
    evaluator.plot_error_histogram()
    evaluator.plot_scatter()
    evaluator.plot_correlation()
