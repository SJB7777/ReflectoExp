from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------
# Helper: 보간하여 모델의 q-grid에 맞추기
# ------------------------------
def resample_spectra(meas_q: np.ndarray, meas_I: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    """
    meas_q: (M,) 실제 측정 q
    meas_I: (M,) 또는 (N_samples, M) 측정 intensity
    target_q: (Q,) 모델 q-grid
    반환: (Q,) 또는 (N_samples, Q)
    """
    # 만약 여러 샘플이면 first axis가 샘플로 오게 정리
    meas_I = np.atleast_2d(meas_I)
    out = np.empty((meas_I.shape[0], target_q.size), dtype=float)
    for i, row in enumerate(meas_I):
        out[i] = np.interp(target_q, meas_q, row, left=row[0], right=row[-1])
    if out.shape[0] == 1:
        return out[0]
    return out

# ------------------------------
# Helper: stats 불러와 파라미터 정규화/역정규화
# ------------------------------
def load_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    st = torch.load(stats_path)
    param_mean = st["param_mean"].numpy() if isinstance(st["param_mean"], torch.Tensor) else np.array(st["param_mean"])
    param_std = st["param_std"].numpy() if isinstance(st["param_std"], torch.Tensor) else np.array(st["param_std"])
    return param_mean, param_std

def normalize_params(params: np.ndarray, param_mean: np.ndarray, param_std: np.ndarray) -> np.ndarray:
    return (params - param_mean) / param_std

def denormalize_params(params_norm: np.ndarray, param_mean: np.ndarray, param_std: np.ndarray) -> np.ndarray:
    return params_norm * param_std + param_mean

# ------------------------------
# Create DataLoader from raw measured arrays
# ------------------------------
def make_dataloader_from_arrays(
    measured_q: np.ndarray,
    measured_I: np.ndarray,        # shape (N, M) or (M,)
    true_params: np.ndarray | None,  # shape (N, n_params) or None
    target_q: np.ndarray,
    stats_path: Path,
    batch_size: int = 64,
    device: torch.device | None = None,
):
    """
    measured_I: can be 1D (single sample) or 2D (N_samples, M)
    true_params: if provided, shape (N, n_params)
    returns: DataLoader that yields (R_tensor, params_norm_tensor_or_dummy)
    """
    # resample to target q-grid
    I_res = resample_spectra(measured_q, measured_I, target_q)
    if I_res.ndim == 1:
        I_res = I_res[None, :]

    # convert to torch tensors
    R_tensor = torch.tensor(I_res, dtype=torch.float32)

    # load stats for parameter normalization
    param_mean, param_std = load_stats(stats_path)

    if true_params is None:
        # dummy targets (required by dataset interface)
        print("실측 파라미터 없음 - 정성적 평가만 진행")
        targets_norm = torch.zeros((R_tensor.shape[0], param_mean.size), dtype=torch.float32)
    else:
        params = np.atleast_2d(true_params)
        params_norm = normalize_params(params, param_mean, param_std)
        targets_norm = torch.tensor(params_norm, dtype=torch.float32)

    dataset = TensorDataset(R_tensor, targets_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, param_mean, param_std

# ------------------------------
# Main: evaluate and additional visualizations
# ------------------------------
def evaluate_with_plots(model, data_loader: DataLoader, stats_path: Path, out_prefix: Path, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()

    st = torch.load(stats_path)
    param_mean = st["param_mean"].to(device) if isinstance(st["param_mean"], torch.Tensor) else torch.tensor(st["param_mean"], device=device)
    param_std = st["param_std"].to(device) if isinstance(st["param_std"], torch.Tensor) else torch.tensor(st["param_std"], device=device)

    preds_norm_list = []
    targets_norm_list = []
    Rs = []

    with torch.no_grad():
        for R_batch, params_norm_batch in data_loader:
            R_batch = R_batch.to(device)
            out = model(R_batch).cpu()
            preds_norm_list.append(out)
            targets_norm_list.append(params_norm_batch)
            Rs.append(R_batch.cpu())

    preds_norm = torch.cat(preds_norm_list, dim=0)
    targets_norm = torch.cat(targets_norm_list, dim=0)
    Rs = torch.cat(Rs, dim=0)

    preds = denormalize_params(preds_norm.numpy(), param_mean.cpu().numpy(), param_std.cpu().numpy())
    targets = denormalize_params(targets_norm.numpy(), param_mean.cpu().numpy(), param_std.cpu().numpy())

    errors = preds - targets
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    n_params = preds.shape[1]
    param_names = ["Thickness (nm)", "Roughness (Å)", "SLD"][:n_params]

    # Print summary
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    for i, name in enumerate(param_names):
        print(f"{name:15s} MAE={mae[i]:.4f} | RMSE={rmse[i]:.4f}")

    # Save parity plots and residual histograms
    out_prefix = Path(out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)

    # parity plots
    for i, name in enumerate(param_names):
        plt.figure(figsize=(5,5))
        plt.scatter(targets[:, i], preds[:, i], s=10)
        mn = min(targets[:, i].min(), preds[:, i].min())
        mx = max(targets[:, i].max(), preds[:, i].max())
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Parity: {name}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        # pfile = out_prefix / f'parity_{i}_{name.replace(" ", "_")}.png'
        # plt.savefig(pfile, dpi=300)
        plt.show()
        plt.close()

    # residual histograms
    for i, name in enumerate(param_names):
        plt.figure(figsize=(6,4))
        plt.hist(errors[:, i], bins=50, density=True, alpha=0.7)
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f'Residual histogram: {name} (MAE={mae[i]:.4f})')
        plt.xlabel('Error')
        plt.ylabel('Density')
        plt.tight_layout()
        # pfile = out_prefix / f'resid_hist_{i}_{name.replace(" ", "_")}.png'
        # plt.savefig(pfile, dpi=300)
        plt.show()
        plt.close()

    # Optional: save per-sample spectrum comparison (measured vs reconstructed simulated from preds)
    # If you have forward simulator, you can reconstruct reflectivity from predicted params and compare.
    # Here we only save raw input R for visual inspection
    np.save(out_prefix / 'preds.npy', preds)
    np.save(out_prefix / 'targets.npy', targets)
    np.save(out_prefix / 'Rs.npy', Rs.numpy())

    print(f"Saved diagnostics to {out_prefix.resolve()}")

    return preds, targets, errors

# ------------------------------
# Example usage block (사용자 환경에 맞게 수정)
# ------------------------------
if __name__ == "__main__":
    from model import XRR1DRegressor
    # -----------------------------
    # 경로 설정
    # -----------------------------
    exp_dir = Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer\run")
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    out_dir = exp_dir / "eval_real"

    # 실제 측정 데이터 디렉토리
    real_data_dir = Path(r'D:\03_Resources\Data\XRR_AI\data\one_layer\real_tester')
    measured_q = np.load(real_data_dir / "measured_q.npy")        # (M,)
    measured_I = np.load(real_data_dir / "measured_I.npy")        # (N, M) 또는 (M,)
    try:
        measured_params = np.load(real_data_dir / "measured_params.npy")  # (N, 3)
    except FileNotFoundError:
        measured_params = None

    # -----------------------------
    # 모델 불러오기
    # -----------------------------
    ckpt = torch.load(checkpoint_file, map_location='cpu')
    config = ckpt['config']['model_args']

    model = XRR1DRegressor(**config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # -----------------------------
    # 학습 때 q-grid 재생성
    # -----------------------------
    q_points = 200
    wavelength = 1.54  # Å
    tth_min = 1.0      # deg
    tth_max = 6.0      # deg

    # θ = tth/2
    tth_deg = np.linspace(tth_min, tth_max, q_points)
    theta = np.deg2rad(tth_deg / 2)
    q_target = 4 * np.pi / wavelength * np.sin(theta)  # Å^-1

    # -----------------------------
    # dataloader 생성
    # -----------------------------
    loader, mean, std = make_dataloader_from_arrays(
        measured_q=measured_q,
        measured_I=measured_I,
        true_params=measured_params,
        target_q=q_target,
        stats_path=stats_file,
        batch_size=64,
    )

    # -----------------------------
    # 평가
    # -----------------------------
    preds, targets, errors = evaluate_with_plots(model, loader, stats_file, out_dir)
