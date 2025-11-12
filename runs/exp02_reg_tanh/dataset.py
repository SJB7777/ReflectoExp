# runs/exp001_reg_tanh/dataset.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


import h5py
import numpy as np
import torch
from torch.serialization import safe_globals
from torch.utils.data import Dataset


class RegressionDatasetH5(Dataset):
    """회귀 전용: 연속값을 [-1, 1] 범위로 정규화"""

    def __init__(
        self,
        h5_path: str,
        mode: str = "train",
        val_ratio: float = 0.2,
    ):
        self.h5_path = Path(h5_path)

        with h5py.File(self.h5_path, "r") as f:
            self.n_total = f["R"].shape[0]

        # Train/Val split
        split_idx = int(self.n_total * (1 - val_ratio))
        self.indices = range(split_idx) if mode == "train" else range(split_idx, self.n_total)

        # Compute stats
        stats_path = self.h5_path.parent / "target_stats.pt"
        if stats_path.exists():
            with safe_globals([np._core.multiarray.scalar]):
                self.stats = torch.load(stats_path, weights_only=False)
        else:
            self.stats = self._compute_stats()
            torch.save(self.stats, stats_path)

        # 실제 데이터의 min/max 범위 (모델에 전달용)
        self.thickness_range = (float(self.stats["th_min"]), float(self.stats["th_max"]))
        self.roughness_range = (float(self.stats["rg_min"]), float(self.stats["rg_max"]))
        self.sld_range = (float(self.stats["sld_min"]), float(self.stats["sld_max"]))

    def _compute_stats(self, sample_n: int = 1000):
        rng = np.random.default_rng(42)
        with h5py.File(self.h5_path, "r") as f:
            idx = rng.choice(self.n_total, min(sample_n, self.n_total), replace=False)
            idx.sort()
            thick = f["thickness"][idx].ravel()
            rough = f["roughness"][idx].ravel()
            sld = f["sld"][idx].ravel()

        return {
            "th_min": thick.min(), "th_max": thick.max(),
            "rg_min": rough.min(), "rg_max": rough.max(),
            "sld_min": sld.min(), "sld_max": sld.max(),
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        with h5py.File(self.h5_path, "r") as f:
            refl = f["R"][real_idx]
            thick = f["thickness"][real_idx]
            rough = f["roughness"][real_idx]
            sld = f["sld"][real_idx]

        # Log scale reflectivity
        refl = np.log10(np.maximum(refl, 1e-15)).astype(np.float32)

        # Normalize targets to [-1, 1] (same as tanh output range)
        thick_norm = self._normalize(thick, self.thickness_range)
        rough_norm = self._normalize(rough, self.roughness_range)
        sld_norm = self._normalize(sld, self.sld_range)

        targets = torch.stack([
            torch.tensor(thick_norm, dtype=torch.float32),
            torch.tensor(rough_norm, dtype=torch.float32),
            torch.tensor(sld_norm, dtype=torch.float32)
        ], dim=-1)

        return torch.from_numpy(refl), targets

    def _normalize(self, values: np.ndarray, value_range: tuple) -> np.ndarray:
        """[min, max] -> [-1, 1]"""
        min_val, max_val = value_range
        return 2 * (values - min_val) / (max_val - min_val + 1e-12) - 1

    def denormalize(self, normalized: torch.Tensor, param_idx: int) -> torch.Tensor:
        """[-1, 1] -> [min, max]"""
        ranges = [self.thickness_range, self.roughness_range, self.sld_range]
        min_val, max_val = ranges[param_idx]
        return (normalized + 1) / 2 * (max_val - min_val) + min_val
