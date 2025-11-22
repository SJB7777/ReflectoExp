from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from xrr_model import XRR1DRegressor

from reflecto.math_utils import fom_log


def calculate_metrics(preds: np.ndarray, targets: np.ndarray):
    """
    MAE, RMSE, MAPE 등 성능 지표 계산
    """
    # Tensor -> Numpy 변환은 함수 외부에서 처리했다고 가정
    errors = preds - targets
    abs_errors = np.abs(errors)

    # MAE (Mean Absolute Error)
    mae = np.mean(abs_errors, axis=0)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    # MAPE (Mean Absolute Percentage Error) - 0 나누기 방지
    eps = 1e-6
    mape = np.mean(abs_errors / (np.abs(targets) + eps), axis=0) * 100

    return {
        "errors": errors,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "fom": fom_log(targets, preds)
    }

def plot_results(preds, targets, errors, param_names, save_path):
    """
    시각화: 예측값 vs 실제값 (Scatter) 및 오차 분포 (Hist)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Scatter Plot (예측 vs 정답)
    for i, name in enumerate(param_names):
        ax = axes[0, i]
        y_pred = preds[:, i]
        y_true = targets[:, i]

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))

        # 이상적인 선 (y=x)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Ideal')
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='royalblue', label='Data')

        ax.set_title(f"{name}: Pred vs True")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Error Histogram (오차 분포)
    for i, name in enumerate(param_names):
        ax = axes[1, i]
        err_data = errors[:, i]

        mu = np.mean(err_data)
        sigma = np.std(err_data)

        ax.hist(err_data, bins=50, density=True, alpha=0.6, color='seagreen', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(mu, color='orange', linestyle=':', linewidth=2, label=f'Mean: {mu:.2f}')

        ax.set_title(f"{name} Error (Std: {sigma:.2f})")
        ax.set_xlabel("Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"📊 리포트 그래프 저장 완료: {save_path}")
    plt.close()


def export_evaluation_csv(preds, targets, errors, param_names, save_path):
    """
    Evaluation results -> Pandas DataFrame -> CSV Export
    """
    data = {}

    # Organize data column by column
    for i, name in enumerate(param_names):
        # Clean column names (optional: remove units for cleaner headers if needed)
        clean_name = name.split(' (')[0]

        data[f"{clean_name}_True"] = targets[:, i]
        data[f"{clean_name}_Pred"] = preds[:, i]
        data[f"{clean_name}_Error"] = errors[:, i]
        data[f"{clean_name}_AbsErr"] = np.abs(errors[:, i])
    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(save_path, index=False, encoding='utf-8-sig') # utf-8-sig handles Greek symbols like Å correctly in Excel
        print(f"💾 Data Exported: {save_path}")


def run_inference_loop(model, test_loader, device):
    """모델 추론 루프 (Normalization 상태의 출력 반환)"""
    model.eval()
    all_preds = []
    all_targets = []

    print(f"평가 루프 실행 중 (Device: {device})...")
    with torch.no_grad():
        # inputs: [Batch, 2, Length]
        # targets: [Batch, 3] (Normalized)
        for inputs, targets in test_loader:
            inputs = inputs.to(device)

            preds = model(inputs) # (Batch, 3)

            all_preds.append(preds.cpu())
            all_targets.append(targets)

    return torch.cat(all_preds), torch.cat(all_targets)

def evaluate_pipeline(
    test_loader: DataLoader,
    checkpoint_path: Path | str,
    stats_path: Path | str,
    report_file: Path | str | None = None,
    report_csv: Path | str | None = None
):
    """
    전체 평가 파이프라인
    로드 -> 추론 -> 역정규화 -> 지표계산 -> 시각화
    """
    checkpoint_path = Path(checkpoint_path)
    stats_path = Path(stats_path)

    # 1. 파일 확인
    if not checkpoint_path.exists() or not stats_path.exists():
        print("❌ 체크포인트 또는 통계 파일이 없습니다.")
        return

    # 2. 장치 및 통계 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = torch.load(stats_path, map_location='cpu')
    param_mean = stats["param_mean"]
    param_std = stats["param_std"]

    # 3. 모델 로드
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get('config', {}).get('model_args', {})

    # 호환성 처리
    if 'input_channels' not in model_args:
        model_args['input_channels'] = 2

    model = XRR1DRegressor(**model_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ 모델 로드 완료: {checkpoint_path.name}")

    # 4. 추론 수행 (Normalized 상태)
    preds_norm, targets_norm = run_inference_loop(model, test_loader, device)

    # 5. 역정규화 (Denormalization)
    # X_real = X_norm * Std + Mean
    preds_real = preds_norm * param_std + param_mean
    targets_real = targets_norm * param_std + param_mean

    # Tensor -> Numpy
    preds_np = preds_real.numpy()
    targets_np = targets_real.numpy()

    # 6. 지표 계산 및 출력
    metrics = calculate_metrics(preds_np, targets_np)

    param_names = ["Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)"]

    print("\n" + "="*65)
    print(f"{'Parameter':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 65)
    for i, name in enumerate(param_names):
        print(f"{name:<20} | {metrics['mae'][i]:<10.4f} | {metrics['rmse'][i]:<10.4f} | {metrics['mape'][i]:<10.2f}")
    print("="*65)

    # 7. 시각화
    if report_file:
        plot_results(preds_np, targets_np, metrics['errors'], param_names, report_file)
    if report_csv:
        export_evaluation_csv(preds_np, targets_np, metrics['errors'], param_names, report_csv)
# =========================================================================
# 실행부 (Main)
# =========================================================================
if __name__ == "__main__":
    from config import CONFIG
    from dataset import XRR1LayerDataset

    # 경로 설정
    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    report_file_img = exp_dir / "evaluation_report.png"
    report_file_csv = exp_dir / "evaluation_results.csv"

    if h5_file.exists():
        # 테스트 데이터셋 로드
        test_set = XRR1LayerDataset(
            h5_file, stats_file, mode="test",
            q_min=CONFIG["simulation"]["q_min"],
            q_max=CONFIG["simulation"]["q_max"],
            n_points=CONFIG["simulation"]["q_points"]
        )

        test_loader = DataLoader(
            test_set, batch_size=CONFIG["training"]["batch_size"],
            shuffle=False, num_workers=0
        )

        # 평가 실행
        evaluate_pipeline(test_loader, checkpoint_file, stats_file, report_file_img, report_file_csv)
    else:
        print("테스트할 데이터셋 파일이 없습니다.")
