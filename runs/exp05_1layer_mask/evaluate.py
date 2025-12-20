from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import XRR1LayerDataset
from xrr_model import XRR1DRegressor
from reflecto_exp.math_utils import powerspace


# ------------------------------------------------------------
# 1. Metrics Calculation
# ------------------------------------------------------------
def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calculate performance metrics (MAE, RMSE, MAPE).
    """
    errors = preds - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors, axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    eps = 1e-6
    mape = np.mean(abs_errors / (np.abs(targets) + eps), axis=0) * 100

    return {
        "errors": errors,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


# ------------------------------------------------------------
# 2. Results DataFrame Construction
# ------------------------------------------------------------
def generate_results_df(
    preds: np.ndarray,
    targets: np.ndarray,
    errors: np.ndarray,
    param_names: list[str]
) -> pd.DataFrame:

    data = {}
    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]

        data[f"{clean}_True"] = targets[:, i]
        data[f"{clean}_Pred"] = preds[:, i]
        data[f"{clean}_Error"] = errors[:, i]

    return pd.DataFrame(data)


# ------------------------------------------------------------
# 3. Save CSV
# ------------------------------------------------------------
def save_results_csv(df: pd.DataFrame, save_path: Path) -> None:

    if not save_path:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Saved CSV report: {save_path}")


# ------------------------------------------------------------
# 4. Save Evaluation Plot
# ------------------------------------------------------------
def save_results_plot(df: pd.DataFrame, param_names: list[str], save_path: Path) -> None:

    if not save_path:
        return

    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(6 * n_params, 10))

    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]

        y_true = df[f"{clean}_True"].values
        y_pred = df[f"{clean}_Pred"].values
        err = df[f"{clean}_Error"].values

        # Scatter
        ax_s = axes[0][i]
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())

        ax_s.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, alpha=0.7)
        ax_s.scatter(y_true, y_pred, alpha=0.3, s=10, color='royalblue')

        ax_s.set_title(f"{name}: Pred vs True")
        ax_s.set_xlabel("True")
        ax_s.set_ylabel("Predicted")
        ax_s.grid(True, alpha=0.3)

        # Histogram
        ax_h = axes[1][i]
        mu = np.mean(err)
        sigma = np.std(err)

        ax_h.hist(err, bins=50, density=True, alpha=0.6,
                  color='seagreen', edgecolor='black')
        ax_h.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax_h.axvline(mu, color='orange', linestyle=':', linewidth=2)

        ax_h.set_title(f"{name} Error (Std: {sigma:.2f})")
        ax_h.set_xlabel("Error")
        ax_h.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Graph report: {save_path}")


# ------------------------------------------------------------
# 5. Save Training History Plot
# ------------------------------------------------------------
def save_history_plot(history: dict, save_path: Path) -> None:

    if not save_path or not history:
        return

    epochs = range(1, len(history['train']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train'], label="Train", lw=2)
    ax1.plot(epochs, history['val'], label="Val", lw=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if "lr" in history:
        ax2.plot(epochs, history['lr'], label="LR", linestyle="--")
        ax2.set_yscale("log")
        ax2.set_title("Learning Rate")
        ax2.set_xlabel("Epochs")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved History Plot: {save_path}")


# ------------------------------------------------------------
# 6. Inference
# ------------------------------------------------------------
def run_inference(model, loader, device):

    model.eval()
    preds, targets = [], []

    print(f"Running inference on {device}...")

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)

            preds.append(pred.cpu())
            targets.append(y)

    return torch.cat(preds), torch.cat(targets)


# ------------------------------------------------------------
# 7. Print Summary Metrics
# ------------------------------------------------------------
def print_metrics_table(metrics: dict[str, np.ndarray], param_names: list[str]) -> None:

    print("\n" + "=" * 65)
    print(f"{'Parameter':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 65)

    for i, name in enumerate(param_names):
        print(f"{name:<20} | "
              f"{metrics['mae'][i]:<10.4f} | "
              f"{metrics['rmse'][i]:<10.4f} | "
              f"{metrics['mape'][i]:<10.2f}")

    print("=" * 65)


# ------------------------------------------------------------
# 8. Evaluation Pipeline
# ------------------------------------------------------------
def evaluate_pipeline(
    test_loader: DataLoader,
    checkpoint_path: Path,
    stats_path: Path,
    report_img_path: Path | None = None,
    report_csv_path: Path | None = None,
    report_history_path: Path | None = None
):

    if not checkpoint_path.exists() or not stats_path.exists():
        print("Missing checkpoint or stats file.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = torch.load(stats_path, map_location="cpu")
    param_mean = stats["param_mean"]
    param_std = stats["param_std"]

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get("config", {}).get("model_args", {})
    history = ckpt.get("history", {})

    model = XRR1DRegressor(**model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded: {checkpoint_path.name}")

    preds_norm, targets_norm = run_inference(model, test_loader, device)

    preds_real = preds_norm * param_std + param_mean
    targets_real = targets_norm * param_std + param_mean

    preds_np = preds_real.numpy()
    targets_np = targets_real.numpy()

    metrics = calculate_metrics(preds_np, targets_np)

    param_names = [
        "Thickness (Å)",
        "Roughness (Å)",
        "SLD (10⁻⁶ Å⁻²)",
    ]

    df = generate_results_df(preds_np, targets_np, metrics["errors"], param_names)

    print_metrics_table(metrics, param_names)

    if report_csv_path:
        save_results_csv(df, report_csv_path)

    if report_img_path:
        save_results_plot(df, param_names, report_img_path)

    if report_history_path:
        save_history_plot(history, report_history_path)


# ------------------------------------------------------------
# 9. Main Entry
# ------------------------------------------------------------
def main():

    base = Path(CONFIG["base_dir"])
    if not base.exists():
        raise FileNotFoundError(f"base_dir not found: {base}")

    exp_dir = base / CONFIG["exp_name"]

    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    ckpt_file = exp_dir / "best.pt"

    report_img = exp_dir / "evaluation_report.png"
    report_csv = exp_dir / "evaluation_results.csv"
    report_history = exp_dir / "training_history.png"

    if not h5_file.exists():
        print("Dataset not found.")
        return

    qs = powerspace(
        CONFIG["simulation"]["q_min"],
        CONFIG["simulation"]["q_max"],
        CONFIG["simulation"]["q_points"],
        CONFIG["simulation"]["power"]
    )

    test_set = XRR1LayerDataset(qs, h5_file, stats_file, mode="test")
    loader = DataLoader(
        test_set,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    evaluate_pipeline(
        test_loader=loader,
        checkpoint_path=ckpt_file,
        stats_path=stats_file,
        report_img_path=report_img,
        report_csv_path=report_csv,
        report_history_path=report_history
    )


if __name__ == "__main__":
    main()
