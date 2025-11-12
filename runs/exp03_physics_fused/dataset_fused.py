# runs/exp04_physics_fused/dataset_fused.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FusedDatasetH5(Dataset):
    def __init__(self, h5_path: str, mode: str = "train", val_ratio: float = 0.2):
        self.h5_path = Path(h5_path)

        with h5py.File(self.h5_path, "r") as f:
            self.n_total = f["R"].shape[0]
            self.q_values = torch.tensor(f["q"][:], dtype=torch.float32)

        # Split
        split_idx = int(self.n_total * (1 - val_ratio))
        self.indices = range(split_idx) if mode == "train" else range(split_idx, self.n_total)

        # Stats
        stats_path = self.h5_path.parent / "target_stats.pt"
        if stats_path.exists():
            self.stats = torch.load(stats_path)
        else:
            self.stats = self._compute_stats()
            torch.save(self.stats, stats_path)

        # Ranges
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
            "th_min": float(thick.min()), "th_max": float(thick.max()),
            "rg_min": float(rough.min()), "rg_max": float(rough.max()),
            "sld_min": float(sld.min()), "sld_max": float(sld.max()),
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

        refl = np.log10(np.maximum(refl, 1e-15)).astype(np.float32)
        thick_norm = self._normalize(thick, self.thickness_range)
        rough_norm = self._normalize(rough, self.roughness_range)
        sld_norm = self._normalize(sld, self.sld_range)

        targets = torch.stack([
            torch.tensor(thick_norm, dtype=torch.float32),
            torch.tensor(rough_norm, dtype=torch.float32),
            torch.tensor(sld_norm, dtype=torch.float32)
        ], dim=-1)

        return {
            "reflectivity": torch.from_numpy(refl),
            "targets": targets,
            "q_values": self.q_values
        }

    def _normalize(self, values: np.ndarray, value_range: tuple) -> np.ndarray:
        min_val, max_val = value_range
        return 2 * (values - min_val) / (max_val - min_val + 1e-12) - 1
