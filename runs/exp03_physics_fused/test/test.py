import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from scipy import signal, fftpack
from dataset_fused import FusedDatasetH5
from physics import PhysicsLayer


class PhysicsValidator:
    def __init__(self, h5_path: str | Path, norm_ranges: dict = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 데이터셋 로드 (train/val 모두)
        self.dataset = FusedDatasetH5(
            h5_path,
            mode="val",  # 검증셋으로 테스트
            val_ratio=0.2,
            norm_ranges=norm_ranges
        )

        # q_values 추출
        with h5py.File(h5_path, "r") as f:
            self.q_values = torch.tensor(f["q"][:], dtype=torch.float32)

        # PhysicsLayer 초기화
        self.physics_layer = PhysicsLayer(
            q_values=self.q_values,
            n_layers=self.dataset[0]["targets"].shape[0],
            confidence_threshold=0.1
        ).to(self.device)

        # 역정규화를 위한 범위
        self.norm_ranges = norm_ranges or {
            "thickness": [0.0, 200.0],
            "roughness": [0.0, 10.0],
            "sld": [0.0, 140.0]
        }

    def denormalize_thickness(self, thickness_norm: np.ndarray) -> np.ndarray:
        """정규화된 두께를 원래 스케일로 복원"""
        min_val, max_val = self.norm_ranges["thickness"]
        return (thickness_norm + 1) / 2 * (max_val - min_val) + min_val

    def validate_sample(self, idx: int, plot: bool = False, save_path: str = None):
        """단일 샘플에 대해 physics layer 검증"""
        sample = self.dataset[idx]

        # 실제 값
        real_targets = sample["targets"]  # (n_layers, 3)
        real_thickness = self.denormalize_thickness(real_targets[:, 0].numpy())

        # physics layer 추정
        reflectivity = sample["reflectivity"].unsqueeze(0).to(self.device)  # (1, Q)

        with torch.no_grad():
            physics_output, validity_mask = self.physics_layer(reflectivity)

        # 역정규화
        est_thickness = physics_output[0].cpu().numpy()  # physics layer는 이미 실제 스케일
        confidence = validity_mask[0].item()

        # 오차 계산
        mae = np.mean(np.abs(est_thickness[:len(real_thickness)] - real_thickness))

        results = {
            "idx": idx,
            "real_thickness": real_thickness,
            "est_thickness": est_thickness,
            "confidence": confidence,
            "mae": mae,
            "is_valid": confidence > 0.1
        }

        # 시각화
        if plot:
            self.plot_physics_process(
                sample,
                physics_output.cpu(),
                validity_mask.cpu(),
                save_path=save_path
            )

        return results

    def validate_batch(self, n_samples: int = 100):
        """배치 단위로 검증 (통계 계산)"""
        indices = np.random.choice(len(self.dataset), min(n_samples, len(self.dataset)), replace=False)

        all_mae = []
        confidences = []
        success_rates = []

        print(f"🧪 Validating {len(indices)} samples...")

        for idx in tqdm(indices, desc="Validating"):
            result = self.validate_sample(idx, plot=False)
            all_mae.append(result["mae"])
            confidences.append(result["confidence"])
            success_rates.append(result["is_valid"])

        # 통계 계산
        valid_mae = [mae for mae, valid in zip(all_mae, success_rates) if valid]
        invalid_mae = [mae for mae, valid in zip(all_mae, success_rates) if not valid]

        stats = {
            "total_samples": len(indices),
            "valid_samples": sum(success_rates),
            "success_rate": np.mean(success_rates),
            "avg_confidence": np.mean(confidences),
            "overall_mae": np.mean(all_mae),
            "valid_mae": np.mean(valid_mae) if valid_mae else 0.0,
            "invalid_mae": np.mean(invalid_mae) if invalid_mae else 0.0,
            "mae_std": np.std(all_mae)
        }

        return stats, indices, all_mae

    def plot_physics_process(self, sample, physics_output, validity_mask, save_path: str = None):
        """Physics layer의 중간 과정 시각화"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        q_np = self.q_values.numpy()
        R_log_np = sample["reflectivity"].numpy()
        R_original = 10 ** R_log_np

        # 1. 원본 반사율
        axes[0].plot(q_np, R_original, 'b-', label='Reflectivity (R)')
        axes[0].set_xlabel('q')
        axes[0].set_ylabel('R')
        axes[0].set_title('Original Reflectivity')
        axes[0].legend()
        axes[0].set_yscale('log')

        # 2. FFT 결과 및 피크 검출
        R_smooth = signal.savgol_filter(R_log_np, window_length=min(len(R_log_np), 51), polyorder=4, mode='interp')
        R_savgol = R_original / (10 ** R_smooth)
        R_savgol = R_savgol / (R_savgol[0] + 1e-12) * R_original.max()

        N = len(R_savgol)
        w = signal.windows.hamming(N)
        yf = 2 / N * np.abs(fftpack.fft(w * R_savgol / np.mean(w), n=10000))
        xf = fftpack.fftfreq(10000, d=q_np[1]-q_np[0])
        xf, yf = xf[:5000], yf[:5000]
        x_fft = xf * 2 * np.pi

        axes[1].plot(x_fft, yf, 'g-', label='FFT')

        # 피크 강조
        from scipy.signal import argrelmax
        upper_idx = np.searchsorted(x_fft, 200.0)
        for order in [50, 30, 20, 10]:
            idx_max = argrelmax(yf[:upper_idx], order=order)[0]
            if len(idx_max) > 0:
                axes[1].plot(x_fft[idx_max], yf[idx_max], 'ro', label=f'Peaks (order={order})')
                break

        axes[1].set_xlabel('Frequency (1/thickness)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('FFT Analysis & Peak Detection')
        axes[1].legend()

        # 3. 두께 비교
        real_thickness = self.denormalize_thickness(sample["targets"][:, 0].numpy())
        est_thickness = physics_output[0].cpu().numpy()

        layers = np.arange(len(real_thickness))
        width = 0.35

        axes[2].bar(layers - width/2, real_thickness, width, label='Real', alpha=0.8)
        axes[2].bar(layers + width/2, est_thickness[:len(real_thickness)], width,
                   label='Physics Est.', alpha=0.8)

        axes[2].set_xlabel('Layer Index')
        axes[2].set_ylabel('Thickness (Å)')
        axes[2].set_title(f'Thickness Comparison (Confidence: {validity_mask[0].item():.3f})')
        axes[2].legend()

        # MAE 표시
        mae = np.mean(np.abs(est_thickness[:len(real_thickness)] - real_thickness))
        axes[2].text(0.02, 0.98, f'MAE: {mae:.2f} Å',
                    transform=axes[2].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def print_summary(self, stats: dict):
        """검증 결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 Physics Layer Validation Summary")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Valid predictions: {stats['valid_samples']} ({stats['success_rate']:.2%})")
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
        print(f"\nThickness MAE (all samples): {stats['overall_mae']:.2f} Å")
        if stats['valid_mae'] > 0:
            print(f"Thickness MAE (valid only): {stats['valid_mae']:.2f} Å")
        if stats['invalid_mae'] > 0:
            print(f"Thickness MAE (invalid only): {stats['invalid_mae']:.2f} Å")
        print(f"MAE std: {stats['mae_std']:.2f} Å")
        print("="*60)

    def plot_error_distribution(self, indices, mae_list, save_path: str | Path = None):
        """오차 분포 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. MAE 히스토그램
        axes[0].hist(mae_list, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('MAE (Å)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Thickness MAE Distribution')
        axes[0].axvline(np.mean(mae_list), color='r', linestyle='--', label=f'Mean: {np.mean(mae_list):.2f}')
        axes[0].legend()

        # 2. Confidence vs MAE 산점도
        confidences = []
        for idx in indices:
            sample = self.dataset[idx]
            reflectivity = sample["reflectivity"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, validity_mask = self.physics_layer(reflectivity)
            confidences.append(validity_mask[0].item())

        axes[1].scatter(confidences, mae_list, alpha=0.6)
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('MAE (Å)')
        axes[1].set_title('Confidence vs MAE')
        axes[1].axvline(0.1, color='r', linestyle='--', label='Threshold (0.1)')
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    n_samples = 200
    h5_path = Path(r"D:\data\XRR_AI\xrr_data.h5")
    plot_dir = Path(".")
    plot_samples = 10
    # 검증기 초기화
    validator = PhysicsValidator(
        h5_path=h5_path,
        norm_ranges={"thickness": [0.0, 200.0], "roughness": [0.0, 10.0], "sld": [0.0, 140.0]}
    )

    # 결과 저장 디렉토리
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)

    # 배치 검증
    stats, indices, mae_list = validator.validate_batch(n_samples)
    validator.print_summary(stats)

    # 오차 분포 플롯
    validator.plot_error_distribution(
        indices, mae_list,
        save_path=plot_dir / "error_distribution.png"
    )

    # 샘플별 상세 플롯
    print(f"\nPlotting {plot_samples} sample details...")
    plot_indices = np.random.choice(indices, plot_samples, replace=False)

    for idx in plot_indices:
        validator.validate_sample(
            idx,
            plot=True,
            save_path=plot_dir / f"sample_{idx}_analysis.png"
        )

    print(f"\nValidation complete! Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
