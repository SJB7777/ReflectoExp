from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from xrr_model import XRR1DRegressor


def evaluate_model(model, test_loader: DataLoader, stats_path: Path | str,
                   report_file: Path | str | None = None, save_plot: bool = True):
    """
    모델 평가 파이프라인: 추론 -> 역정규화 -> 지표 계산 -> 시각화
    """
    stats_path = Path(stats_path)
    device = next(model.parameters()).device

    # 1. 통계 정보 로드 (역정규화용)
    if not stats_path.exists():
        raise FileNotFoundError(f"통계 파일을 찾을 수 없습니다: {stats_path}")

    stats = torch.load(stats_path, map_location=device)
    param_mean = stats["param_mean"] # Shape: (3,)
    param_std = stats["param_std"]   # Shape: (3,)

    # 2. 추론 (Inference)
    model.eval()
    all_preds_norm = []
    all_targets_norm = []

    print(f"평가 시작 (Device: {device})...")

    with torch.no_grad():
        # inputs: [Batch, 2, Length] (LogR, Mask)
        # targets_norm: [Batch, 3] (Normalized Params)
        for inputs, targets_norm in test_loader:
            inputs = inputs.to(device)
            targets_norm = targets_norm.to(device)

            # Forward
            preds_norm = model(inputs)

            all_preds_norm.append(preds_norm.cpu())
            all_targets_norm.append(targets_norm.cpu())

    # 3. 데이터 합치기 및 역정규화 (Denormalization)
    # Shape: (N_samples, 3)
    preds_norm_tensor = torch.cat(all_preds_norm)
    targets_norm_tensor = torch.cat(all_targets_norm)

    # CPU로 이동하여 역정규화 수행 (X_real = X_norm * Std + Mean)
    # param_mean/std도 CPU로 이동
    param_mean = param_mean.cpu()
    param_std = param_std.cpu()

    preds = preds_norm_tensor * param_std + param_mean
    targets = targets_norm_tensor * param_std + param_mean

    # 4. 성능 지표 계산
    # 오차 = 예측 - 정답
    errors = preds - targets
    abs_errors = torch.abs(errors)

    # MAE (Mean Absolute Error)
    mae = torch.mean(abs_errors, dim=0)
    # RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(torch.mean(errors ** 2, dim=0))
    # MAPE (Mean Absolute Percentage Error) - 0으로 나누는 것 방지
    mape = torch.mean(abs_errors / (torch.abs(targets) + 1e-6), dim=0) * 100

    # 5. 결과 출력
    param_names = ["Thickness (Å)", "Roughness (Å)", "SLD (1e-6 Å⁻²)"]

    print("\n" + "="*60)
    print(f"{'Parameter':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 60)
    for i, name in enumerate(param_names):
        print(f"{name:<20} | {mae[i]:<10.4f} | {rmse[i]:<10.4f} | {mape[i]:<10.2f}")
    print("="*60)

    # 6. 시각화 (옵션)
    if save_plot and report_file:
        plot_results(preds.numpy(), targets.numpy(), errors.numpy(), param_names, report_file)

    return {
        "mae": mae.numpy(),
        "rmse": rmse.numpy(),
        "preds": preds.numpy(),
        "targets": targets.numpy()
    }

def plot_results(preds, targets, errors, param_names, save_path):
    """평가 결과 시각화 (Scatter Plot + Histogram)"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Predicted vs Actual (Scatter Plot)
    for i, name in enumerate(param_names):
        ax = axes[0, i]
        y_pred = preds[:, i]
        y_true = targets[:, i]

        # 이상치 제외하고 그리기 위해 범위 설정 (선택사항)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='royalblue', label='Samples')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

        ax.set_title(f"{name} : Pred vs True")
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Row 2: Error Distribution (Histogram)
    for i, name in enumerate(param_names):
        ax = axes[1, i]
        err_data = errors[:, i]

        # 통계치
        mu = np.mean(err_data)
        sigma = np.std(err_data)

        ax.hist(err_data, bins=50, density=True, alpha=0.6, color='seagreen', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(mu, color='orange', linestyle=':', linewidth=2, label=f'Mean: {mu:.2f}')

        # 타이틀에 표준편차(정밀도) 표시
        ax.set_title(f"{name} Error Dist.\n(Std: {sigma:.2f})")
        ax.set_xlabel("Error (Pred - True)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"평가 그래프 저장 완료: {save_path}")
    plt.close()

def load_checkpoint_and_evaluate(test_loader: DataLoader, checkpoint_path: Path | str,
                                 stats_path: Path | str, report_file: Path | str):
    """체크포인트를 로드하여 모델 복원 후 평가 수행"""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"체크포인트 파일 없음: {checkpoint_path}")
        return

    # 1. 체크포인트 로드
    # map_location을 사용하여 CUDA 모델을 CPU 환경에서도 로드 가능하게 함
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 2. Config 복원
    # 저장된 모델 아규먼트가 있으면 사용, 없으면 기본값 (여기서는 main.py에서 저장했다고 가정)
    model_args = ckpt.get('config', {}).get('model_args', {})

    # 호환성: 이전 코드와 config 구조가 다를 경우를 대비한 기본값 설정
    # Dataset에서 2채널(LogR, Mask)을 주므로 input_channels=2는 필수
    if 'input_channels' not in model_args:
        model_args['input_channels'] = 2

    print(f"모델 복원 설정: {model_args}")

    # 3. 모델 생성 및 가중치 로드
    model = XRR1DRegressor(**model_args)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    # 4. 평가 실행
    evaluate_model(model, test_loader, stats_path, report_file)

if __name__ == "__main__":
    # 테스트용 실행 코드
    from config import CONFIG
    from dataset import XRR1LayerDataset

    base_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]

    h5_file = base_dir / "dataset.h5"
    stats_file = base_dir / "stats.pt"
    checkpoint_file = base_dir / "best.pt"
    report_file = base_dir / "test_report.png"

    if h5_file.exists():
        # Dataset 생성
        test_set = XRR1LayerDataset(
            h5_file, stats_file, mode="test",
            q_min=CONFIG["simulation"]["q_min"],
            q_max=CONFIG["simulation"]["q_max"],
            n_points=CONFIG["simulation"]["q_points"]
        )

        test_loader = DataLoader(test_set, batch_size=CONFIG["training"]["batch_size"],
                                 shuffle=False, num_workers=0)

        load_checkpoint_and_evaluate(test_loader, checkpoint_file, stats_file, report_file)
    else:
        print("테스트할 데이터셋이 없습니다.")
