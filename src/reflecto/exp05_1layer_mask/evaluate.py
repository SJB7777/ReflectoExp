from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reflecto.math_utils import fom_log

from .config import CONFIG
from .dataset import XRR1LayerDataset
from .xrr_model import XRR1DRegressor


def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calculate performance metrics (MAE, RMSE, MAPE).
    """
    errors = preds - targets
    abs_errors = np.abs(errors)

    # MAE (Mean Absolute Error)
    mae = np.mean(abs_errors, axis=0)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    # MAPE (Mean Absolute Percentage Error) - avoid division by zero
    eps = 1e-6
    mape = np.mean(abs_errors / (np.abs(targets) + eps), axis=0) * 100

    return {
        "errors": errors,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "fom": fom_log(targets, preds)
    }


def generate_results_df(
    preds: np.ndarray,
    targets: np.ndarray,
    errors: np.ndarray,
    param_names: list[str]
) -> pd.DataFrame:
    """
    Organize raw numpy arrays into a structured Pandas DataFrame.
    Structure: [Param_True, Param_Pred, Param_Error] for each parameter.
    """
    data = {}
    for i, name in enumerate(param_names):
        # Clean name for CSV headers (remove units)
        clean_name = name.split(' (')[0]

        data[f"{clean_name}_True"] = targets[:, i]
        data[f"{clean_name}_Pred"] = preds[:, i]
        data[f"{clean_name}_Error"] = errors[:, i]

    return pd.DataFrame(data)


def save_results_csv(df: pd.DataFrame, save_path: Path) -> None:
    """
    Save the results DataFrame to a CSV file.
    """
    if not save_path:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig ensures special characters (Angstrom, Greek letters) open correctly in Excel
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Saved CSV report: {save_path}")


def save_results_plot(
    df: pd.DataFrame,
    param_names: list[str],
    save_path: Path
) -> None:
    """
    Visualize results from the DataFrame and save as an image.
    Generates Scatter plots (Pred vs True) and Error Histograms.
    """
    if not save_path:
        return

    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(6 * n_params, 10))

    for i, name in enumerate(param_names):
        clean_name = name.split(' (')[0]
        col_true = f"{clean_name}_True"
        col_pred = f"{clean_name}_Pred"
        col_err = f"{clean_name}_Error"

        y_true = df[col_true].values
        y_pred = df[col_pred].values
        err_data = df[col_err].values

        # 1. Scatter Plot
        ax_scatter = axes[0, i]
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))

        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Ideal')
        ax_scatter.scatter(y_true, y_pred, alpha=0.3, s=10, color='royalblue', label='Data')

        ax_scatter.set_title(f"{name}: Pred vs True")
        ax_scatter.set_xlabel("True Value")
        ax_scatter.set_ylabel("Predicted Value")
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)

        # 2. Error Histogram
        ax_hist = axes[1, i]
        mu = np.mean(err_data)
        sigma = np.std(err_data)

        ax_hist.hist(err_data, bins=50, density=True, alpha=0.6, color='seagreen', edgecolor='black')
        ax_hist.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax_hist.axvline(mu, color='orange', linestyle=':', linewidth=2, label=f'Mean: {mu:.2f}')

        ax_hist.set_title(f"{name} Error (Std: {sigma:.2f})")
        ax_hist.set_xlabel("Error")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved Graph report: {save_path}")
    plt.close()


def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute model inference on the given loader.
    Returns (Predictions, Targets) in Normalized Tensor format.
    """
    model.eval()
    all_preds = []
    all_targets = []

    print(f"Running inference loop on {device}...")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            preds = model(inputs)

            all_preds.append(preds.cpu())
            all_targets.append(targets)

    return torch.cat(all_preds), torch.cat(all_targets)


def print_metrics_table(metrics: dict[str, np.ndarray], param_names: list[str]) -> None:
    """
    Print a formatted table of metrics to the console.
    """
    print("\n" + "=" * 65)
    print(f"{'Parameter':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 65)

    for i, name in enumerate(param_names):
        mae = metrics['mae'][i]
        rmse = metrics['rmse'][i]
        mape = metrics['mape'][i]
        print(f"{name:<20} | {mae:<10.4f} | {rmse:<10.4f} | {mape:<10.2f}")

    print("=" * 65)


def evaluate_pipeline(
    test_loader: DataLoader,
    checkpoint_path: Path,
    stats_path: Path,
    report_img_path: Path | None = None,
    report_csv_path: Path | None = None
) -> None:
    """
    Orchestrates the full evaluation process:
    Load -> Inference -> Denormalize -> Calculate -> Save.
    """
    # 1. Validation
    if not checkpoint_path.exists() or not stats_path.exists():
        print(f"Error: Missing checkpoint ({checkpoint_path}) or stats file ({stats_path}).")
        return

    # 2. Setup Resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Statistics
    stats = torch.load(stats_path, map_location='cpu')
    param_mean = stats["param_mean"]
    param_std = stats["param_std"]

    # Load Model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get('config', {}).get('model_args', {})
    if 'input_channels' not in model_args:
        model_args['input_channels'] = 2
    if "output_dim" not in model_args:
        model_args['output_dim'] = 3

    model = XRR1DRegressor(**model_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model loaded: {checkpoint_path.name}")

    # 3. Inference (Normalized)
    preds_norm, targets_norm = run_inference(model, test_loader, device)

    # 4. Denormalization
    # Formula: X_real = X_norm * Std + Mean
    preds_real = preds_norm * param_std + param_mean
    targets_real = targets_norm * param_std + param_mean

    # Convert to Numpy
    preds_np = preds_real.numpy()
    targets_np = targets_real.numpy()

    # 5. Metrics & Data Construction
    metrics = calculate_metrics(preds_np, targets_np)

    # Define parameter names (ensure this matches model output dimension)
    param_names = ["Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)"]

    # Create Master DataFrame
    results_df = generate_results_df(preds_np, targets_np, metrics['errors'], param_names)

    # 6. Output & Reporting
    print_metrics_table(metrics, param_names)

    if report_csv_path:
        save_results_csv(results_df, report_csv_path)

    if report_img_path:
        save_results_plot(results_df, param_names, report_img_path)


def main():
    # Configuration & Paths
    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"

    report_file_img = exp_dir / "evaluation_report.png"
    report_file_csv = exp_dir / "evaluation_results.csv"

    if not h5_file.exists():
        print("Dataset file not found. Cannot proceed.")
        return

    # Dataset Preparation
    test_set = XRR1LayerDataset(
        h5_file, stats_file, mode="test",
        q_min=CONFIG["simulation"]["q_min"],
        q_max=CONFIG["simulation"]["q_max"],
        n_points=CONFIG["simulation"]["q_points"]
    )

    test_loader = DataLoader(
        test_set,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # Execute Pipeline
    evaluate_pipeline(
        test_loader=test_loader,
        checkpoint_path=checkpoint_file,
        stats_path=stats_file,
        report_img_path=report_file_img,
        report_csv_path=report_file_csv
    )


if __name__ == "__main__":
    main()
