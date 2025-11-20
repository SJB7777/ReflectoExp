from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class XRR1LayerDataset(Dataset):
    """1-layer 전용 Dataset (Reflectivity max-normalization + log10 고정)"""

    def __init__(self, h5_file: str | Path, stats_file: str | Path,
                 mode: str = "train", val_ratio: float = 0.2):

        self.h5_path = Path(h5_file)
        self.stats_path = stats_file
        self.mode = mode

        with h5py.File(self.h5_path, "r") as hf:
            self.q_values = hf["q"][:]
            self.thickness = hf["thickness"][:].squeeze()
            self.roughness = hf["roughness"][:].squeeze()
            self.sld = hf["sld"][:].squeeze()
            self.reflectivity = hf["R"][:]

        self.n_total = len(self.reflectivity)

        # 명확한 Train/Val/Test split
        test_ratio = 0.1
        train_ratio = 1.0 - val_ratio - test_ratio

        train_end = int(self.n_total * train_ratio)
        val_end = int(self.n_total * (train_ratio + val_ratio))

        if mode == "train":
            self.indices = range(train_end)
        elif mode == "val":
            self.indices = range(train_end, val_end)
        else:
            self.indices = range(val_end, self.n_total)

        self._setup_normalization()

    # ---------------------------------------------------------
    # 파라미터 정규화 (reflectivity는 max-normalization만 적용하므로 통계 불필요)
    # ---------------------------------------------------------
    def _setup_normalization(self):

        if self.mode == "train":
            train_indices = range(int(self.n_total * 0.7))

            # reflectivity는 max-normalization만 하므로 평균/분산 없음

            params = np.stack([
                self.thickness[train_indices],
                self.roughness[train_indices],
                self.sld[train_indices]
            ], axis=1)

            self.param_mean = np.mean(params, axis=0)
            self.param_std = np.std(params, axis=0)

            torch.save({
                "param_mean": torch.from_numpy(self.param_mean),
                "param_std": torch.from_numpy(self.param_std)
            }, self.stats_path)

            print(f"정규화 파라미터 저장: {self.stats_path}")

        else:
            stats = torch.load(self.stats_path)
            self.param_mean = stats["param_mean"].numpy()
            self.param_std = stats["param_std"].numpy()

    def __len__(self):
        return len(self.indices)

    # ---------------------------------------------------------
    # Reflectivity max-normalization + log10
    # ---------------------------------------------------------
    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        R = self.reflectivity[real_idx]

        R_max = np.max(R)  # 물리적 측면의 intensity scale 제거
        R_scaled = R / (R_max + 1e-15)

        R_log = np.log10(np.maximum(R_scaled, 1e-15))
        R_tensor = torch.from_numpy(R_log).float()

        # ------------------ params -------------------
        params = np.array([
            self.thickness[real_idx],
            self.roughness[real_idx],
            self.sld[real_idx]
        ], dtype=np.float32)

        params_norm = (params - self.param_mean) / self.param_std
        params_tensor = torch.from_numpy(params_norm).float()

        return R_tensor, params_tensor

    # ---------------------------------------------------------
    # Parameter 역정규화
    # ---------------------------------------------------------
    def denormalize_params(self, params_norm):
        if isinstance(params_norm, torch.Tensor):
            params_norm = params_norm.detach().cpu().numpy()
        params = params_norm * self.param_std + self.param_mean
        return params

    # ---------------------------------------------------------
    # 실측 reflectivity 정규화 (로그 고정)
    # ---------------------------------------------------------
    def normalize_reflectivity(self, R_real: np.ndarray):
        R_max = np.max(R_real)
        R_scaled = R_real / (R_max + 1e-15)
        R_log = np.log10(np.maximum(R_scaled, 1e-15))
        return torch.from_numpy(R_log).float(), R_max

    # ---------------------------------------------------------
    # 역정규화: logR → R × original_max
    # ---------------------------------------------------------
    @staticmethod
    def denormalize_reflectivity(R_log: np.ndarray, original_max: float):
        R = 10 ** R_log
        return R * original_max
