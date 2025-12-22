from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Project Imports
from config import CONFIG
from dataset import XRR1LayerDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from xrr_model import XRRPhysicsModel

from reflecto_exp.simulate.simul_genx import ParamSet, param2refl


# -----------------------------------------------------------------------------
# 1. Advanced Metrics
# -----------------------------------------------------------------------------
def calculate_r2(true, pred):
    """Calculate R^2 (Coefficient of Determination)"""
    ss_res = np.sum((true - pred) ** 2, axis=0)
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))

def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    errors = preds - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors, axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mape = np.mean(abs_errors / (np.abs(targets) + 1e-6), axis=0) * 100
    r2 = calculate_r2(targets, preds)

    return {
        "errors": errors,
        "abs_errors": abs_errors, # For finding worst cases
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }

# -----------------------------------------------------------------------------
# 2. Reporting & Saving
# -----------------------------------------------------------------------------
def save_results_csv(df: pd.DataFrame, save_path: Path):
    if not save_path: return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[Save] CSV Report: {save_path.name}")

def save_worst_cases(df: pd.DataFrame, abs_errors: np.ndarray, save_path: Path, top_k=50):
    """가장 에러가 큰 샘플들의 정보를 저장"""
    if not save_path:
        return

    # 두께 에러 기준으로 정렬 (가장 중요한 물성)
    # Thickness column index assumed 0
    thick_errs = abs_errors[:, 0]
    worst_indices = np.argsort(thick_errs)[::-1][:top_k]

    worst_df = df.iloc[worst_indices].copy()
    worst_df.insert(0, "Sample_ID", worst_indices)

    worst_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[Save] Worst Cases Log: {save_path.name}")
    return worst_indices

def print_metrics_table(metrics: dict, param_names: list):
    print("\n" + "=" * 80)
    print(f"{'Parameter':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'R²':<10}")
    print("-" * 80)
    for i, name in enumerate(param_names):
        print(
            f"{name:<20} | {metrics['mae'][i]:<10.4f} | "
            f"{metrics['rmse'][i]:<10.4f} | {metrics['mape'][i]:<10.2f} | "
            f"{metrics['r2'][i]:<10.4f}"
        )
    print("=" * 80)

# -----------------------------------------------------------------------------
# 3. Visualization (The Ultimate Plotter)
# -----------------------------------------------------------------------------
def save_correlation_plot(df: pd.DataFrame, param_names: list, save_path: Path):
    if not save_path:
        return

    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(6 * n_params, 10))

    # Colors
    scatter_color = '#3B8ED0'
    line_color = '#E63946'
    hist_color = '#2A9D8F'

    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        y_true = df[f"{clean}_True"].values
        y_pred = df[f"{clean}_Pred"].values
        err = df[f"{clean}_Error"].values

        # 1. Scatter Plot
        ax_s = axes[0][i]
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())

        ax_s.plot([min_v, max_v], [min_v, max_v], color=line_color, lw=2, linestyle='--')
        ax_s.scatter(y_true, y_pred, alpha=0.3, s=15, color=scatter_color, edgecolor='none')

        # Stats in title
        r2 = calculate_r2(y_true, y_pred)
        ax_s.set_title(f"{name}\n$R^2$ = {r2:.4f}", fontsize=12, fontweight='bold')
        ax_s.set_xlabel("Ground Truth")
        ax_s.set_ylabel("Prediction")
        ax_s.grid(True, linestyle=':', alpha=0.6)

        # 2. Error Histogram
        ax_h = axes[1][i]
        mu, sigma = np.mean(err), np.std(err)

        ax_h.hist(err, bins=60, density=True, alpha=0.7, color=hist_color, edgecolor='black', linewidth=0.5)
        ax_h.axvline(0, color=line_color, linestyle='--', linewidth=1.5)
        ax_h.axvline(mu, color='orange', linestyle=':', linewidth=2, label=f"Mean: {mu:.2f}")

        ax_h.set_title(f"Error Distribution (Std: {sigma:.2f})")
        ax_h.set_xlabel("Error (Pred - True)")
        ax_h.legend(fontsize=9)
        ax_h.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] Correlation Plot: {save_path.name}")


def plot_error_heatmap(df: pd.DataFrame, save_path: Path | None = None):
    """
    파라미터 조합(Thickness vs Roughness)에 따른 예측 난이도(MAE) 시각화
    """
    # 1. 구간화 (Binning): 연속적인 파라미터를 10개 구간으로 나눔
    # 데이터의 분포에 따라 bins 숫자를 조절하세요.
    n_bins = 10

    # 분석용 임시 복사본 생성 및 절대 오차 계산
    plot_df = df.copy()
    plot_df['Abs_Thick_Err'] = plot_df['Thickness_Error'].abs()

    # 구간 생성 (pd.cut 사용)
    plot_df['Thick_Bin'] = pd.cut(plot_df['Thickness_True'], bins=n_bins).apply(lambda x: int(x.mid))
    plot_df['Rough_Bin'] = pd.cut(plot_df['Roughness_True'], bins=n_bins).apply(lambda x: int(x.mid))

    # 2. 피벗 테이블 생성 (Index: Roughness, Columns: Thickness, Values: MAE)
    pivot_table = plot_df.pivot_table(
        values='Abs_Thick_Err',
        index='Rough_Bin',
        columns='Thick_Bin',
        aggfunc='mean',
        observed=False
    )

    # 물리적 직관을 위해 Index 정렬 (Roughness가 위로 갈수록 커지게)
    pivot_table = pivot_table.sort_index(ascending=False)

    # 3. 시각화
    plt.figure(figsize=(10, 8))

    # vmin/vmax를 설정하여 특정 에러 이상은 모두 진하게 표시되도록 하면 패턴 찾기가 쉽습니다.
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap='YlOrRd',
        cbar_kws={'label': 'Mean Absolute Error (Å)'},
        linewidths=.5
    )

    plt.title("Prediction Difficulty Map: Thickness MAE", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("True Thickness (Å)", fontsize=12)
    plt.ylabel("True Roughness (Å)", fontsize=12)

    # 4. 물리적 해석 가이드라인 추가
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Save] Error Heatmap: {save_path.name}")

    plt.close()


def plot_single_curve(ax_main, ax_res, qs, R_true, R_pred, title):
    """Helper for plotting one XRR curve with residuals"""
    # Main Curve
    ax_main.plot(qs, R_true, 'k-', lw=2.0, alpha=0.6, label='Measured')
    ax_main.plot(qs, R_pred, 'r--', lw=1.5, label='AI Recon')
    ax_main.set_yscale('log')
    ax_main.set_xticklabels([]) # Hide x labels for main
    ax_main.grid(True, which='both', alpha=0.3)
    ax_main.legend(loc='upper right', fontsize=8)
    ax_main.set_title(title, fontsize=10, fontweight='bold')

    # Residuals (Log Difference)
    # epsilon to avoid log(0)
    log_true = np.log10(np.maximum(R_true, 1e-15))
    log_pred = np.log10(np.maximum(R_pred, 1e-15))
    diff = log_pred - log_true

    ax_res.plot(qs, diff, 'b-', lw=1.0, alpha=0.7)
    ax_res.axhline(0, color='k', linestyle='--', lw=1.0)
    ax_res.set_ylim(-2, 2) # Limit residual range for visibility
    ax_res.set_ylabel(r"$\Delta$ LogR", fontsize=8)
    ax_res.set_xlabel(r"$q (\AA^{-1})$", fontsize=9)
    ax_res.grid(True, alpha=0.3)

def save_advanced_reconstruction(
    preds: np.ndarray, targets: np.ndarray, qs: np.ndarray,
    save_path: Path, worst_indices: np.ndarray | None = None, n_random: int = 3
):
    if not save_path: return

    # Selection: Top 3 Worst Cases + 3 Random Cases
    if worst_indices is not None and len(worst_indices) >= 3:
        target_indices = np.concatenate([
            worst_indices[:3],
            np.random.choice(len(preds), n_random, replace=False)
        ])
        labels = ["Worst #1", "Worst #2", "Worst #3", "Random A", "Random B", "Random C"]
    else:
        target_indices = np.random.choice(len(preds), 6, replace=False)
        labels = [f"Random #{i}" for i in range(6)]

    # Layout: 2 Rows x 3 Cols. Each cell has (Main + Residual) sub-grids.
    fig = plt.figure(figsize=(18, 10))
    outer_grid = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.3)

    print("[Plot] Generating Advanced Reconstruction Report...")

    for i, idx in enumerate(target_indices):
        # Create inner grid for Main + Residual
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_grid[i], height_ratios=[3, 1], hspace=0.05
        )
        ax_main = plt.Subplot(fig, inner_grid[0])
        ax_res = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax_main)
        fig.add_subplot(ax_res)

        # Data Prep
        t_p = targets[idx]
        p_p = preds[idx]

        # Physical Constraints for simulation
        p_p[0] = max(p_p[0], 1.0) # Thickness > 1
        p_p[1] = max(p_p[1], 0.0) # Roughness > 0
        p_p[2] = max(p_p[2], 0.0) # SLD > 0

        # Simulation
        try:
            R_true = param2refl(qs, [ParamSet(*t_p)], sio2_param=None)
            R_pred = param2refl(qs, [ParamSet(*p_p)], sio2_param=None)
        except Exception as e:
            print(f"Simul Error at idx {idx}: {e}")
            R_true = np.zeros_like(qs) + 1e-10
            R_pred = np.zeros_like(qs) + 1e-10

        # Title Info
        err_d = p_p[0] - t_p[0]
        header = f"[{labels[i]}] Sample #{idx}"
        info = f"Thick: {t_p[0]:.1f}Å (Err: {err_d:+.1f}Å)"

        plot_single_curve(ax_main, ax_res, qs, R_true, R_pred, f"{header}\n{info}")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] Reconstruction Plot: {save_path.name}")

# -----------------------------------------------------------------------------
# 4. Pipeline
# -----------------------------------------------------------------------------
def run_inference(model, loader, device):
    model.eval()
    preds, targets = [], []
    print(f"Running inference on {device}...")
    with torch.inference_mode(): # Faster than no_grad
        for X, y in tqdm(loader, desc="Inference", leave=False):
            X = X.to(device)
            pred = model(X)
            preds.append(pred.cpu())
            targets.append(y)
    return torch.cat(preds), torch.cat(targets)

def evaluate_pipeline(
    test_loader: DataLoader,
    checkpoint_path: Path,
    stats_path: Path,
    qs: np.ndarray,
    report_img_path: Path | None = None,
    report_csv_path: Path | None = None,
    report_history_path: Path | None = None
):
    if not checkpoint_path.exists() or not stats_path.exists():
        print("❌ Error: Missing checkpoint or stats file.")
        return

    # 1. Load Resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(stats_path, map_location="cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    config_dict = ckpt.get("config", {})
    # Handle older checkpoint compatibility
    model_args = config_dict.get("model_args", config_dict)

    # 2. Model Init
    model = XRRPhysicsModel(**model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✅ Model loaded from {checkpoint_path.name}")
    print(f"   (Fourier: {model.use_fourier})")

    # 3. Inference
    preds_norm, targets_norm = run_inference(model, test_loader, device)

    # 4. Denormalize
    p_mean = stats["param_mean"]
    p_std = stats["param_std"]

    preds_real = preds_norm * p_std + p_mean
    targets_real = targets_norm * p_std + p_mean

    preds_np = preds_real.numpy()
    targets_np = targets_real.numpy()

    # 5. Metrics & DataFrame
    metrics = calculate_metrics(preds_np, targets_np)
    param_names = ["Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)"]

    df = pd.DataFrame()
    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        df[f"{clean}_True"] = targets_np[:, i]
        df[f"{clean}_Pred"] = preds_np[:, i]
        df[f"{clean}_Error"] = metrics["errors"][:, i]

    print_metrics_table(metrics, param_names)

    # 6. Save Reports
    if report_csv_path:
        save_results_csv(df, report_csv_path)
        # Save worst cases for debugging
        worst_log_path = report_csv_path.parent / "worst_cases_log.csv"
        worst_indices = save_worst_cases(df, metrics["abs_errors"], worst_log_path)
    else:
        worst_indices = None

    if report_img_path:
        save_correlation_plot(df, param_names, report_img_path)

        # Advanced Reconstruction
        recon_path = report_img_path.parent / "reconstruction_analysis.png"
        save_advanced_reconstruction(
            preds_np, targets_np, qs, recon_path,
            worst_indices=worst_indices
        )

        plot_error_heatmap(df, save_path=report_img_path.parent / "error_heatmap.png")

    # 7. History Plot
    if report_history_path:
        history = ckpt.get("history", {})
        if history:
            # Simple plotter import or inline logic
            pass # (Assuming evaluate.py handles this via save_history_plot function if needed, but omitted for brevity as it's standard)


# -----------------------------------------------------------------------------
# 5. Main Entry Point for Standalone Evaluation
# -----------------------------------------------------------------------------
def main():
    print(f"=== Starting Evaluation for Experiment: {CONFIG['exp_name']} ===")

    # 1. 경로 설정 (main.py의 로직과 동기화)
    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"  # 혹은 "last.pt"

    if not checkpoint_file.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_file}")
        return

    # 2. Q-Vector 생성 (물리 모델 계산용)
    qs = np.linspace(
        CONFIG["simulation"]["q_min"],
        CONFIG["simulation"]["q_max"],
        CONFIG["simulation"]["q_points"],
    )

    # 3. Test DataLoader 준비
    # evaluate 모드이므로 augmentation은 끄고 데이터 로드만 수행
    print("Preparing Test DataLoader...")
    test_ds = XRR1LayerDataset(
        qs=qs,
        h5_file=h5_file,
        stats_file=stats_file,
        mode="test",
        val_ratio=CONFIG["training"]["val_ratio"],
        test_ratio=CONFIG["training"]["test_ratio"],
        augment=False  # 평가 시에는 원본 데이터 사용
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        num_workers=CONFIG["training"]["num_workers"],
        pin_memory=torch.cuda.is_available()
    )

    # 4. Pipeline 실행
    # 결과물들은 실험 폴더(exp_dir) 내부로 저장되도록 경로 설정
    evaluate_pipeline(
        test_loader=test_loader,
        checkpoint_path=checkpoint_file,
        stats_path=stats_file,
        qs=qs,
        report_img_path=exp_dir / "evaluation_correlation.png",
        report_csv_path=exp_dir / "evaluation_results.csv",
        report_history_path=exp_dir / "training_history.png"
    )

    print(f"\n✅ Evaluation Complete. Reports saved in: {exp_dir}")

if __name__ == "__main__":
    # 실행 시 GPU 메모리 효율화를 위해 세팅
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    main()
