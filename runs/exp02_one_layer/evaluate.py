from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import XRR1DRegressor
from torch.utils.data import DataLoader


def evaluate(model, test_loader: DataLoader, stats_path: Path | str, report_file: Path | None = None, save_plot: bool = True):
    """
    모델 평가: 역정규화 후 물리적 오류 계산
    """
    stats_path = Path(stats_path)
    device = next(model.parameters()).device

    # 정규화 파라미터 로드
    stats = torch.load(stats_path)
    param_mean = stats["param_mean"]
    param_std = stats["param_std"]

    model.eval()
    all_preds_norm = []
    all_targets_norm = []

    print("테스트 평가 중...")
    with torch.no_grad():
        for R, params_norm in test_loader:
            R = R.to(device)
            pred_norm = model(R).cpu()

            all_preds_norm.append(pred_norm)
            all_targets_norm.append(params_norm)

    # 역정규화
    preds = torch.cat(all_preds_norm) * param_std + param_mean
    targets = torch.cat(all_targets_norm) * param_std + param_mean

    # 오류 계산
    errors = preds - targets
    mae = torch.mean(torch.abs(errors), dim=0)
    rmse = torch.sqrt(torch.mean(errors ** 2, dim=0))

    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    param_names = ["Thickness (nm)", "Roughness (Å)", "SLD"]
    for i, name in enumerate(param_names):
        print(f"{name:15s}: MAE={mae[i]:7.3f} | RMSE={rmse[i]:7.3f}")
    print("="*50)

    # 시각화
    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, name in enumerate(param_names):
            ax = axes[i]
            data = errors[:, i].numpy()

            ax.hist(data, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.axvline(np.mean(data), color='orange', linestyle=':', label=f'Mean: {np.mean(data):.3f}')

            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Density')
            ax.set_title(f'{name}\nMAE={mae[i]:.3f}')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_file, dpi=300, bbox_inches='tight')
        print(f"오류 분포 그래프 저장: {report_file}")

    return errors, mae, rmse


def load_checkpoint_and_evaluate(test_loader: DataLoader, checkpoint_path: Path, stats_path: Path, report_file: Path):
    """체크포인트 로드 및 평가"""
    # 체크포인트 로드
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = ckpt['config']['model_args']

    # 모든 인자로 모델 생성
    model = XRR1DRegressor(**config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 평가
    evaluate(model, test_loader, stats_path, report_file)


if __name__ == "__main__":
    from config import CONFIG
    from dataset import XRR1LayerDataset

    exp_dir = Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer\run")
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    report_file = exp_dir / "error_distribution.png"

    test_set = XRR1LayerDataset(h5_file, stats_file, mode="test", val_ratio=CONFIG["training"]["val_ratio"])
    test_loader = DataLoader(test_set, batch_size=CONFIG["training"]["batch_size"],
    shuffle=False, num_workers=0)

    load_checkpoint_and_evaluate(test_loader, checkpoint_file, stats_file, report_file)
