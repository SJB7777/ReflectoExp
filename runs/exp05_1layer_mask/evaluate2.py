from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# === User Modules ===
from config import CONFIG
from dataset import XRR1LayerDataset
from torch.utils.data import DataLoader
from tqdm import tqdm  # 진행상황 표시용
from xrr_model import XRR1DRegressor

from reflecto.math_utils import powerspace
from reflecto.simulate.simul_genx import ParamSet, param2refl

# 스타일 설정
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
# 0. FOM Calculation Logic
# ------------------------------------------------------------
def fom_log(ref_exp, ref_calc, epsilon=1e-10):
    """
    Calculates Figure of Merit (log-scale difference).
    Lower is better.
    """
    # log(0) 방지
    ref_exp = np.maximum(ref_exp, epsilon)
    ref_calc = np.maximum(ref_calc, epsilon)
    return np.mean(np.abs(np.log10(ref_exp) - np.log10(ref_calc)))

def calculate_all_foms(qs, preds, targets):
    """
    전체 데이터셋에 대해 Curve를 생성하고 FOM을 계산합니다.
    """
    foms = []
    print("Calculating FOM for all test samples (Physics Simulation)...")
    
    # 배치 처리가 가능하다면 더 빠르겠지만, 안전하게 루프로 처리
    for i in tqdm(range(len(preds))):
        p_pred = ParamSet(preds[i][0], preds[i][1], preds[i][2])
        p_true = ParamSet(targets[i][0], targets[i][1], targets[i][2])

        # param2refl returns (1, len(qs)) usually, need flatten
        curve_pred = param2refl(qs, [p_pred]).flatten()
        curve_true = param2refl(qs, [p_true]).flatten()

        val = fom_log(curve_true, curve_pred)
        foms.append(val)

    return np.array(foms)

# ------------------------------------------------------------
# 1. Metrics & Data Utils (Existing)
# ------------------------------------------------------------
def calculate_metrics(preds, targets):
    errors = preds - targets
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors, axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    eps = 1e-6
    mape = np.mean(abs_errors / (np.abs(targets) + eps), axis=0) * 100
    return {"errors": errors, "mae": mae, "rmse": rmse, "mape": mape}

def generate_results_df(preds, targets, errors, foms, param_names):
    data = {}
    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        data[f"{clean}_True"] = targets[:, i]
        data[f"{clean}_Pred"] = preds[:, i]
        data[f"{clean}_Error"] = errors[:, i]
    
    # FOM 컬럼 추가
    data["FOM_Log"] = foms
    return pd.DataFrame(data)

# ------------------------------------------------------------
# 2. FOM Distribution Plot (New!)
# ------------------------------------------------------------
def save_fom_report(foms: np.ndarray, save_path: Path):
    """
    FOM 분포를 히스토그램과 통계로 보여주는 보고서
    """
    if not save_path: return
    
    fig = plt.figure(figsize=(10, 6), layout='constrained')
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1])
    
    # 1. Histogram & KDE
    ax1 = fig.add_subplot(gs[0])
    # 이상치 제외하고 보여주기 위해 99% quantile까지만 자름 (그래프 가독성)
    cut_off = np.percentile(foms, 99)
    foms_view = foms[foms <= cut_off]
    
    ax1.hist(foms_view, bins=30, color='royalblue', alpha=0.7, density=True, label='Density')
    ax1.set_title("FOM Distribution (Log Scale Error)", fontweight='bold')
    ax1.set_xlabel("FOM Value (Lower is Better)")
    ax1.set_ylabel("Density")
    
    # 평균, 중위수 표시
    mean_val = np.mean(foms)
    median_val = np.median(foms)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
    ax1.axvline(median_val, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
    ax1.legend()

    # 2. Summary Stats Text Box
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
    
    ax2.text(0.1, 0.5, stats_text, fontsize=13, family='monospace', 
             verticalalignment='center', transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved FOM Report: {save_path}")

# ------------------------------------------------------------
# 3. Curve Reconstruction Plot (Updated with FOM)
# ------------------------------------------------------------
def save_curve_comparison(qs, preds, targets, foms, save_path: Path):
    if not save_path: return

    # FOM 기준으로 Best/Worst 정렬 (MSE 대신 FOM 사용이 더 설득력 있음)
    sorted_indices = np.argsort(foms)
    
    indices = np.concatenate([
        sorted_indices[:2], # Best FOM
        np.random.choice(sorted_indices[2:-2], 2, replace=False), # Random
        sorted_indices[-2:] # Worst FOM
    ])
    
    titles = ["Best Case 1", "Best Case 2", "Random Case 1", "Random Case 2", "Worst Case 1", "Worst Case 2"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout='constrained')
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        t_true = targets[idx]
        t_pred = preds[idx]
        fom_val = foms[idx] # 해당 샘플의 FOM
        
        p_true = ParamSet(t_true[0], t_true[1], t_true[2])
        p_pred = ParamSet(t_pred[0], t_pred[1], t_pred[2])
        
        curve_true = param2refl(qs, [p_true]).flatten()
        curve_pred = param2refl(qs, [p_pred]).flatten()
        
        ax.semilogy(qs, curve_true, 'k-', lw=3, alpha=0.6, label='Ground Truth')
        ax.semilogy(qs, curve_pred, 'r--', lw=2, label='AI Prediction')
        
        # 제목에 FOM 표시 (핵심)
        title_str = f"{titles[i]}\nFOM: {fom_val:.5f} | Thick Err: {abs(t_true[0]-t_pred[0]):.1f}Å"
        
        # FOM이 좋으면(낮으면) 초록색, 나쁘면 빨간색 텍스트
        color = 'darkgreen' if fom_val < 0.05 else 'black'
        if i >= 4: color = 'firebrick' # Worst cases
            
        ax.set_title(title_str, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel("Q ($Å^{-1}$)")
        
        if i % 3 == 0: ax.set_ylabel("Reflectivity (log)")
        if i == 0: ax.legend(frameon=True)
        ax.grid(True, which='both', alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Curve Proof: {save_path}")

# ------------------------------------------------------------
# 4. Investor Report (Metrics - Same as before)
# ------------------------------------------------------------
def save_investor_report_plots(df: pd.DataFrame, param_names: list[str], save_path: Path):
    if not save_path: return
    n_params = len(param_names)
    fig = plt.figure(figsize=(18, 12), layout='constrained')
    gs = gridspec.GridSpec(3, n_params, figure=fig)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        y_true = df[f"{clean}_True"].values
        y_pred = df[f"{clean}_Pred"].values
        err = df[f"{clean}_Error"].values
        
        # Hexbin
        ax1 = fig.add_subplot(gs[0, i])
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax1.hexbin(y_true, y_pred, gridsize=40, cmap='Blues', mincnt=1, edgecolors='none')
        ax1.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, alpha=0.8)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        ax1.text(0.05, 0.95, f'$R^2 = {corr**2:.3f}$', transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax1.set_title(f"{clean}: Pred vs True", fontweight='bold')
        
        # Residuals
        ax2 = fig.add_subplot(gs[1, i])
        ax2.scatter(y_true, err, alpha=0.4, s=15, color=colors[i % len(colors)], edgecolor='w', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='--')
        std_err = np.std(err)
        ax2.axhline(1.96 * std_err, color='red', linestyle=':', alpha=0.6)
        ax2.axhline(-1.96 * std_err, color='red', linestyle=':', alpha=0.6)
        ax2.set_title(f"{clean}: Residuals", fontweight='bold')
        
        # CDF
        ax3 = fig.add_subplot(gs[2, i])
        sorted_abs_err = np.sort(np.abs(err))
        p = 1. * np.arange(len(err)) / (len(err) - 1)
        ax3.plot(sorted_abs_err, p, lw=3, color='darkslategray')
        idx_90 = int(len(err) * 0.9)
        val_90 = sorted_abs_err[idx_90]
        ax3.axvline(val_90, color='firebrick', linestyle='--')
        ax3.text(val_90, 0.5, f' 90% < {val_90:.2f}', color='firebrick', fontweight='bold')
        ax3.set_title(f"{clean}: Error CDF", fontweight='bold')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Metric Report: {save_path}")

# ------------------------------------------------------------
# 5. Pipeline
# ------------------------------------------------------------
def evaluate_pipeline(test_loader, checkpoint_path, stats_path, qs, output_dir):

    if not checkpoint_path.exists():
        print("Checkpoint not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Stats & Model
    stats = torch.load(stats_path, map_location="cpu")
    param_mean = stats["param_mean"]
    param_std = stats["param_std"]

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = XRR1DRegressor(**ckpt["config"]["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Model loaded. Running Inference...")

    preds_list, targets_list = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X)
            preds_list.append(pred.cpu())
            targets_list.append(y)

    preds_norm = torch.cat(preds_list)
    targets_norm = torch.cat(targets_list)

    # Denormalize
    preds_real = (preds_norm * param_std + param_mean).numpy()
    targets_real = (targets_norm * param_std + param_mean).numpy()

    # --- FOM Calculation (NEW) ---
    # 모든 샘플에 대해 시뮬레이션 및 FOM 계산
    foms = calculate_all_foms(qs, preds_real, targets_real)

    # Metrics
    metrics = calculate_metrics(preds_real, targets_real)
    param_names = ["Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)"]

    # 1. Save CSV (with FOM)
    df = generate_results_df(preds_real, targets_real, metrics["errors"], foms, param_names)
    df.to_csv(output_dir / "evaluation_results_with_fom.csv", index=False)

    # 2. Plots
    save_investor_report_plots(df, param_names, output_dir / "investor_metrics_report.png")

    # 3. FOM Report (NEW Graph)
    save_fom_report(foms, output_dir / "investor_fom_stats.png")

    # 4. Curve Proof with FOM Titles (Updated)
    save_curve_comparison(qs, preds_real, targets_real, foms, output_dir / "investor_curve_proof.png")

    # Console Output
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 60)
    print(f"{'Mean FOM':<20} | {np.mean(foms):.5f}")
    print(f"{'Median FOM':<20} | {np.median(foms):.5f}")
    print("="*60)

# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------
def main():
    base = Path(CONFIG["base_dir"])
    exp_dir = base / CONFIG["exp_name"]

    qs = powerspace(
        CONFIG["simulation"]["q_min"],
        CONFIG["simulation"]["q_max"],
        CONFIG["simulation"]["q_points"],
        CONFIG["simulation"]["power"]
    )

    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    ckpt_file = exp_dir / "best.pt"

    if not h5_file.exists():
        print("Dataset not found.")
        return

    test_set = XRR1LayerDataset(qs, h5_file, stats_file, mode="test")
    loader = DataLoader(
        test_set,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False
    )

    evaluate_pipeline(
        test_loader=loader,
        checkpoint_path=ckpt_file,
        stats_path=stats_file,
        qs=qs,
        output_dir=exp_dir
    )

if __name__ == "__main__":
    main()