from pathlib import Path
import argparse  # [New] 커맨드라인 인자 처리를 위해 추가
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# === User Modules ===
from config import CONFIG
from dataset import XRR1LayerDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from xrr_model import XRR1DRegressor

from reflecto.math_utils import powerspace
from reflecto.simulate.simul_genx import ParamSet, param2refl

# ------------------------------------------------------------
# 0. Global Style Settings
# ------------------------------------------------------------
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150
})

# ------------------------------------------------------------
# 1. Calculation Logic (Metrics & FOM)
# ------------------------------------------------------------
def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    기본적인 회귀 성능 지표 계산 (MAE, RMSE, MAPE)
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

def fom_log(ref_exp, ref_calc, epsilon=1e-10):
    """
    Figure of Merit (Log-scale difference) 계산
    낮을수록 좋음.
    """
    ref_exp = np.maximum(ref_exp, epsilon)
    ref_calc = np.maximum(ref_calc, epsilon)
    return np.mean(np.abs(np.log10(ref_exp) - np.log10(ref_calc)))

def calculate_all_foms(qs, preds, targets):
    """
    AI 예측 파라미터로 물리 시뮬레이션을 돌려 실제 커브와 비교 (FOM 계산)
    """
    foms = []
    print("\n[Physics Check] Calculating FOM for all test samples...")

    # tqdm으로 진행상황 표시
    for i in tqdm(range(len(preds)), desc="Simulating Curves"):
        p_pred = ParamSet(preds[i][0], preds[i][1], preds[i][2])
        p_true = ParamSet(targets[i][0], targets[i][1], targets[i][2])

        # param2refl returns shape (1, len(qs)), flatten required
        curve_pred = param2refl(qs, [p_pred]).flatten()
        curve_true = param2refl(qs, [p_true]).flatten()

        val = fom_log(curve_true, curve_pred)
        foms.append(val)

    return np.array(foms)

def generate_results_df(preds, targets, errors, foms, param_names):
    data = {}
    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        data[f"{clean}_True"] = targets[:, i]
        data[f"{clean}_Pred"] = preds[:, i]
        data[f"{clean}_Error"] = errors[:, i]

    # FOM 컬럼 추가 (foms가 있을 때만)
    if foms is not None:
        data["FOM_Log"] = foms
        
    return pd.DataFrame(data)

# ------------------------------------------------------------
# 2. Plotting Functions
# ------------------------------------------------------------
def save_history_plot(history: dict, save_path: Path) -> None:
    if not save_path or not history:
        return

    epochs = range(1, len(history['train']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
    ax1.plot(epochs, history['train'], label="Train Loss", lw=2)
    ax1.plot(epochs, history['val'], label="Val Loss", lw=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LR Plot
    if "lr" in history:
        ax2.plot(epochs, history['lr'], label="Learning Rate", color='orange', linestyle="--")
        ax2.set_yscale("log")
        ax2.set_title("Learning Rate Schedule")
        ax2.set_xlabel("Epochs")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved History Plot: {save_path}")

def save_investor_report_plots(df: pd.DataFrame, param_names: list[str], save_path: Path):
    if not save_path:
        return

    n_params = len(param_names)
    fig = plt.figure(figsize=(18, 12), layout='constrained')
    gs = gridspec.GridSpec(3, n_params, figure=fig)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        y_true = df[f"{clean}_True"].values
        y_pred = df[f"{clean}_Pred"].values
        err = df[f"{clean}_Error"].values

        # 1. Hexbin Scatter
        ax1 = fig.add_subplot(gs[0, i])
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax1.hexbin(y_true, y_pred, gridsize=40, cmap='Blues', mincnt=1, edgecolors='none')
        ax1.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, alpha=0.8)

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        ax1.text(0.05, 0.95, f'$R^2 = {corr**2:.3f}$', transform=ax1.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax1.set_title(f"{clean}: Pred vs True", fontweight='bold')

        # 2. Residuals Plot
        ax2 = fig.add_subplot(gs[1, i])
        ax2.scatter(y_true, err, alpha=0.4, s=15, color=colors[i % len(colors)], edgecolor='w', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='--')
        std_err = np.std(err)
        ax2.axhline(1.96 * std_err, color='red', linestyle=':', alpha=0.6)
        ax2.axhline(-1.96 * std_err, color='red', linestyle=':', alpha=0.6)
        ax2.set_title(f"{clean}: Residuals", fontweight='bold')

        # 3. CDF
        ax3 = fig.add_subplot(gs[2, i])
        sorted_abs_err = np.sort(np.abs(err))
        p = 1. * np.arange(len(err)) / (len(err) - 1)
        ax3.plot(sorted_abs_err, p, lw=3, color='darkslategray')

        idx_90 = int(len(err) * 0.9)
        val_90 = sorted_abs_err[idx_90]
        ax3.axvline(val_90, color='firebrick', linestyle='--')
        ax3.text(val_90, 0.5, f' 90% < {val_90:.2f}', color='firebrick', fontweight='bold')
        ax3.set_title(f"{clean}: Error CDF", fontweight='bold')
        ax3.set_xlabel("Absolute Error")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Metric Report: {save_path}")

def save_fom_report(foms: np.ndarray, save_path: Path):
    if not save_path or foms is None:
        return

    fig = plt.figure(figsize=(10, 6), layout='constrained')
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1])

    # 1. Histogram
    ax1 = fig.add_subplot(gs[0])
    cut_off = np.percentile(foms, 99)
    foms_view = foms[foms <= cut_off]

    ax1.hist(foms_view, bins=30, color='royalblue', alpha=0.7, density=True, label='Density')
    ax1.set_title("FOM Distribution (Log Scale Error)", fontweight='bold')
    ax1.set_xlabel("FOM Value (Lower is Better)")
    ax1.set_ylabel("Density")

    mean_val = np.mean(foms)
    median_val = np.median(foms)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
    ax1.axvline(median_val, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
    ax1.legend()

    # 2. Stats Text
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    stats_text = (
        f"DATASET SUMMARY\n"
        f"-----------------------\n"
        f"Total Samples : {len(foms):,}\n\n"
        f"FOM Statistics:\n"
        f"  - Mean   : {mean_val:.5f}\n"
        f"  - Median : {median_val:.5f}\n"
        f"  - Min    : {np.min(foms):.5f}\n"
        f"  - Max    : {np.max(foms):.5f}\n\n"
        f"Performance Guarantee:\n"
        f"  - Top 50% < {np.percentile(foms, 50):.4f}\n"
        f"  - Top 90% < {np.percentile(foms, 90):.4f}\n"
        f"  - Top 99% < {np.percentile(foms, 99):.4f}"
    )
    
    ax2.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', 
             verticalalignment='center', transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved FOM Report: {save_path}")

def save_curve_comparison(qs, preds, targets, foms, save_path: Path):
    if not save_path or foms is None: 
        return

    # FOM 기준으로 정렬
    sorted_indices = np.argsort(foms)
    
    # Best 2, Random 2, Worst 2 선택
    indices = np.concatenate([
        sorted_indices[:2], 
        np.random.choice(sorted_indices[2:-2], 2, replace=False), 
        sorted_indices[-2:]
    ])

    titles = ["Best Case 1", "Best Case 2", "Random Case 1", "Random Case 2", "Worst Case 1", "Worst Case 2"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout='constrained')
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]

        t_true = targets[idx]
        t_pred = preds[idx]
        fom_val = foms[idx]

        p_true = ParamSet(t_true[0], t_true[1], t_true[2])
        p_pred = ParamSet(t_pred[0], t_pred[1], t_pred[2])

        curve_true = param2refl(qs, [p_true]).flatten()
        curve_pred = param2refl(qs, [p_pred]).flatten()

        ax.semilogy(qs, curve_true, 'k-', lw=3, alpha=0.6, label='Ground Truth')
        ax.semilogy(qs, curve_pred, 'r--', lw=2, label='AI Prediction')

        thick_err = abs(t_true[0]-t_pred[0])
        title_str = f"{titles[i]}\nFOM: {fom_val:.5f} | Thick Err: {thick_err:.1f}Å"

        color = 'darkgreen' if fom_val < 0.05 else 'black'
        if i >= 4:
            color = 'firebrick' # Worst cases

        ax.set_title(title_str, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel("Q ($Å^{-1}$)")

        if i % 3 == 0:
            ax.set_ylabel("Reflectivity (log)")
        if i == 0:
            ax.legend(frameon=True)
        ax.grid(True, which='both', alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Curve Comparison: {save_path}")

# ------------------------------------------------------------
# 3. Main Evaluation Pipeline
# ------------------------------------------------------------
def evaluate_pipeline(
    test_loader: DataLoader,
    checkpoint_path: Path,
    stats_path: Path,
    qs: np.ndarray,
    output_dir: Path,
    calc_physics_fom: bool = False  # [New Argument]
):

    if not checkpoint_path.exists() or not stats_path.exists():
        print("Error: Missing checkpoint or stats file.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")
    if calc_physics_fom:
        print(">> Option: Physics-based FOM calculation is ENABLED (This may take a while).")
    else:
        print(">> Option: Physics-based FOM calculation is DISABLED.")

    # 1. Load Stats & Model
    stats = torch.load(stats_path, map_location="cpu")
    param_mean = stats["param_mean"]
    param_std = stats["param_std"]

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get("config", {}).get("model_args", {})
    history = ckpt.get("history", {})

    model = XRR1DRegressor(**model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model loaded: {checkpoint_path.name}")

    # 2. Run Inference
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Inference"):
            X = X.to(device)
            pred = model(X)
            preds_list.append(pred.cpu())
            targets_list.append(y)

    preds_norm = torch.cat(preds_list)
    targets_norm = torch.cat(targets_list)

    # 3. Denormalize
    preds_real = (preds_norm * param_std + param_mean).numpy()
    targets_real = (targets_norm * param_std + param_mean).numpy()

    # 4. Calculate Physics FOM (Optional)
    foms = None
    if calc_physics_fom:
        foms = calculate_all_foms(qs, preds_real, targets_real)

    # 5. Calculate Standard Metrics
    metrics = calculate_metrics(preds_real, targets_real)
    param_names = ["Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)"]

    # 6. Console Output Summary
    print("\n" + "=" * 65)
    print(f"{'Parameter':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 65)
    for i, name in enumerate(param_names):
        print(f"{name:<20} | "
              f"{metrics['mae'][i]:<10.4f} | "
              f"{metrics['rmse'][i]:<10.4f} | "
              f"{metrics['mape'][i]:<10.2f}")

    if foms is not None:
        print("-" * 65)
        print(f"Mean FOM: {np.mean(foms):.5f} | Median FOM: {np.median(foms):.5f}")
    print("=" * 65)

    # 7. Save Outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV Report
    df = generate_results_df(preds_real, targets_real, metrics["errors"], foms, param_names)
    df.to_csv(output_dir / "evaluation_results.csv", index=False)
    print(f"Saved CSV Report: {output_dir / 'evaluation_results.csv'}")

    # Plot 1: Training History
    save_history_plot(history, output_dir / "training_history.png")

    # Plot 2: Investor Metrics (Standard)
    save_investor_report_plots(df, param_names, output_dir / "investor_metrics_report.png")

    # Plot 3 & 4 (Only if FOM calculated)
    if foms is not None:
        save_fom_report(foms, output_dir / "investor_fom_stats.png")
        save_curve_comparison(qs, preds_real, targets_real, foms, output_dir / "investor_curve_proof.png")
    else:
        print("Skipping physics-based plots (FOM Stats, Curve Comparison) as FOM was not calculated.")


# ------------------------------------------------------------
# 4. Main Entry
# ------------------------------------------------------------
def main():
    # Argument Parser 추가
    parser = argparse.ArgumentParser(description="Evaluate XRR AI Model")
    parser.add_argument('--fom', action='store_true', help='Calculate physics-based FOM (Slow!)')
    args = parser.parse_args()

    base = Path(CONFIG["base_dir"])
    if not base.exists():
        raise FileNotFoundError(f"base_dir not found: {base}")

    exp_dir = base / CONFIG["exp_name"]

    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    ckpt_file = exp_dir / "best.pt"

    if not h5_file.exists():
        print("Dataset not found.")
        return

    # Q-space generation for Physics Simulation
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
        qs=qs,
        output_dir=exp_dir,
        calc_physics_fom=args.fom
    )


if __name__ == "__main__":
    main()