from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class XRR1LayerDataset(Dataset):
    """1-layer 전용 Dataset"""

    def __init__(self, h5_file: str | Path, stats_file: str | Path, mode: str = "train", val_ratio: float = 0.2):
        self.h5_path = Path(h5_file)
        self.stats_path = stats_file
        self.mode = mode

        with h5py.File(self.h5_path, "r") as hf:
            self.q_values = hf["q"][:]
            self.thickness = hf["thickness"][:].squeeze()
            self.roughness = hf["roughness"][:].squeeze()
            self.sld = hf["sld"][:].squeeze()
            self.reflectivity = hf["R"][:]

        self.n_total = len(self.thickness)

        # 명확한 분할 (Train: 70%, Val: 20%, Test: 10%)
        test_ratio = 0.1
        train_ratio = 1.0 - val_ratio - test_ratio

        train_end = int(self.n_total * train_ratio)
        val_end = int(self.n_total * (train_ratio + val_ratio))

        if mode == "train":
            self.indices = range(train_end)
        elif mode == "val":
            self.indices = range(train_end, val_end)
        else:  # test
            self.indices = range(val_end, self.n_total)

        self._setup_normalization()

    def _setup_normalization(self):
        """정규화 파라미터 설정"""
        if self.mode == "train":
            # 통계 계산 (train 부분만)
            train_indices = range(int(self.n_total * 0.7))

            R_log = np.log10(np.maximum(self.reflectivity[train_indices], 1e-15))

            self.refl_mean = np.mean(R_log, axis=0)
            self.refl_std = np.std(R_log, axis=0)

            params = np.stack([
                self.thickness[train_indices],
                self.roughness[train_indices],
                self.sld[train_indices]
            ], axis=1)

            self.param_mean = np.mean(params, axis=0)
            self.param_std = np.std(params, axis=0)

            # torch tensor로 저장하여 torch.load 문제 해결
            torch.save({
                "refl_mean": torch.from_numpy(self.refl_mean),
                "refl_std": torch.from_numpy(self.refl_std),
                "param_mean": torch.from_numpy(self.param_mean),
                "param_std": torch.from_numpy(self.param_std),
            }, self.stats_path)
            print(f"정규화 파라미터 저장: {self.stats_path}")

        else:
            # 저장된 tensor 로드
            stats = torch.load(self.stats_path)
            self.refl_mean = stats["refl_mean"].numpy()
            self.refl_std = stats["refl_std"].numpy()
            self.param_mean = stats["param_mean"].numpy()
            self.param_std = stats["param_std"].numpy()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # 반사율 처리
        R = self.reflectivity[real_idx]
        R_log = np.log10(np.maximum(R, 1e-15))
        R_norm = (R_log - self.refl_mean) / self.refl_std

        # 파라미터
        params = np.array([
            self.thickness[real_idx],
            self.roughness[real_idx],
            self.sld[real_idx]
        ], dtype=np.float32)

        params_norm = (params - self.param_mean) / self.param_std

        return (
            torch.from_numpy(R_norm).float(),
            torch.from_numpy(params_norm).float()
        )
    def denormalize_params(self, params_norm: torch.Tensor | np.ndarray) -> np.ndarray:
        """모델 출력(정규화된)을 실제 물리 파라미터 스케일로 복원."""
        if isinstance(params_norm, torch.Tensor):
            params_norm = params_norm.detach().cpu().numpy()

        params = params_norm * self.param_std + self.param_mean
        return params
    def normalize_reflectivity(self, R_real: np.ndarray) -> torch.Tensor:
        """실험 R 데이터를 모델 입력용으로 정규화."""
        R_log = np.log10(np.maximum(R_real, 1e-15))

        # 학습 시 저장된 mean/std 사용
        R_norm = (R_log - self.refl_mean) / self.refl_std
        return torch.from_numpy(R_norm).float()
