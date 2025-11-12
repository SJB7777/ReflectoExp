import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FusedDatasetH5(Dataset):
    def __init__(self, h5_path: str | Path,
        mode: str = "train",
        val_ratio: float = 0.2,
        norm_ranges: dict | None = None
    ):
        self.h5_path = Path(h5_path)
        self.mode = mode
        if not self.h5_path.exists():
            raise FileNotFoundError(f"'{self.h5_path}' not found.")

        with h5py.File(self.h5_path, "r") as f:
            self.n_total = f["R"].shape[0]
            self.q_values = torch.tensor(f["q"][:], dtype=torch.float32)

        # Split
        split_idx = int(self.n_total * (1 - val_ratio))
        self.indices = range(split_idx) if mode == "train" else range(split_idx, self.n_total)

        # Ranges
        if norm_ranges is None:
            norm_ranges = {
                "thickness": [0.0, 200.0],
                "roughness" : [0.0, 10.0],
                "sld" : [0.0, 140.0]
            }
        self.thickness_range = norm_ranges["thickness"]
        self.roughness_range = norm_ranges["roughness"]
        self.sld_range = norm_ranges["sld"]

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

    @staticmethod
    def _normalize(values: np.ndarray, value_range: tuple) -> np.ndarray:
        min_val, max_val = value_range
        return 2 * (values - min_val) / (max_val - min_val + 1e-12) - 1
