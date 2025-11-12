import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import RegressionDatasetH5
from model import XRRRegressor


def evaluate(checkpoint_path: str, h5_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Create model
    # Note: q_len and n_layers must match training
    # For demonstration, we load from config saved in checkpoint
    q_len = 100  # Adjust based on your data
    n_layers = 2  # Adjust based on your data

    model = XRRRegressor(q_len=q_len, n_layers=n_layers).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load stats
    stats = ckpt.get("stats", None)
    if stats:
        model.set_target_stats(stats["mean"].to(device), stats["std"].to(device))

    # Dataset
    dataset = RegressionDatasetH5(h5_path, mode="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    # Evaluation
    all_errors = []

    with torch.no_grad():
        for refl, targets in loader:
            refl = refl.to(device)

            outputs = model(refl)
            physical = outputs["physical"]  # (B, L, 3)

            # Calculate error (denormalized)
            # targets are normalized, so we need to denormalize for comparison
            denorm_targets = torch.zeros_like(targets)
            for i in range(3):
                denorm_targets[:, :, i] = dataset.denormalize(targets[:, :, i], i)

            errors = physical.cpu() - denorm_targets.cpu()
            all_errors.append(errors.numpy())

    # Plot
    errors = np.concatenate(all_errors).reshape(-1, 3)  # (N, 3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    param_names = ["Thickness (nm)", "Roughness (Å)", "SLD (1e-6 Å^-2)"]

    for i, name in enumerate(param_names):
        ax = axes[i]
        data = errors[:, i]

        ax.hist(data, bins=40, alpha=0.6, color='blue', edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--',
                   label=f'Mean: {data.mean():.3f}')
        ax.axvline(data.mean() + data.std(), color='green', linestyle=':',
                   label=f'+1σ: {data.mean() + data.std():.3f}')
        ax.axvline(data.mean() - data.std(), color='green', linestyle=':',
                   label=f'-1σ: {data.mean() - data.std():.3f}')

        ax.set_xlabel(f'Prediction Error ({name})')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Error Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("regression_error_distribution.png", dpi=300)
    plt.show()

    # Metrics
    mae = np.mean(np.abs(errors), axis=0)
    print(f"MAE - Thickness: {mae[0]:.4f}, Roughness: {mae[1]:.4f}, SLD: {mae[2]:.4f}")


if __name__ == "__main__":
    ckpt_dir = r"D:\03_Resources\Data\XRR_AI\model\exp001_reg_tanh\best.pt"
    h5_dir = r"D:\03_Resources\Data\XRR_AI\data\p300o6_raw.h5"
    evaluate(ckpt_dir, h5_dir)
