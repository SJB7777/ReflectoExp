# evaluate.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def evaluate(model, test_loader, stats_path: Path, save_plot: bool = True):
    """
    모델 평가: 역정규화 후 물리적 오류 계산
    """
    device = next(model.parameters()).device

    # 정규화 파라미터 로드
    stats = torch.load(stats_path)
    param_mean = stats["param_mean"]  # (3,)
    param_std = stats["param_std"]    # (3,)

    model.eval()
    all_preds_norm = []
    all_targets_norm = []

    with torch.no_grad():
        for R, params_norm in test_loader:
            R = R.to(device)
            pred_norm = model(R).cpu()

            all_preds_norm.append(pred_norm)
            all_targets_norm.append(params_norm)

    # 역정규화 (물리적 값으로 복원)
    preds = torch.cat(all_preds_norm) * param_std + param_mean
    targets = torch.cat(all_targets_norm) * param_std + param_mean

    # 오류 계산
    errors = preds - targets
    mae = torch.mean(torch.abs(errors), dim=0)
    rmse = torch.sqrt(torch.mean(errors ** 2, dim=0))

    print("\n" + "="*50)
    print("📈 TEST RESULTS")
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
        plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
        print("📊 오류 분포 그래프 저장: error_distribution.png")

    return errors, mae, rmse

def load_model_and_evaluate(checkpoint_path: Path, h5_path: Path, stats_path: Path):
    """체크포인트 로드 및 평가"""
    # 모델 구조 복원
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = ckpt['config']['model_args']

    from model import XRR1DRegressor
    model = XRR1DRegressor(q_len=config['q_len'])
    model.load_state_dict(ckpt['model_state_dict'])

    # 테스트 데이터셋
    from dataset import XRR1LayerDataset
    from torch.utils.data import DataLoader

    test_set = XRR1LayerDataset(h5_path, mode="test")
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # 평가
    evaluate(model, test_loader, stats_path)
