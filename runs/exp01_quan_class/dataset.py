"""
Improved Dataset + Quantizer for XRR project.

Changes:
- ParamQuantizer: added state_dict / load_state_dict / fit_from_dataset
- DatasetH5: safe per-worker HDF5 opening, optional log-scaling, normalization, downsampling.
- __getitem__ returns CPU tensors (no device pinned in dataset).
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from reflecto_exp.simulate.simul_refnx import ParamSet


class ParamQuantizer:
    """
    Binning (quantization) utility for thickness / roughness / sld.

    Usage:
      q = ParamQuantizer()
      q.fit_from_dataset(path, n_sample=1000)  # optional, build bins from data
      q.state_dict() / q.load_state_dict(...) for saving/loading
      q.quantize(param) -> np.ndarray([t_idx, r_idx, s_idx])
    """

    def __init__(
        self,
        thickness_bins: np.ndarray | None = None,
        roughness_bins: np.ndarray | None = None,
        sld_bins: np.ndarray | None = None
    ):
        # sensible defaults (units: nm for thickness, Å for roughness, 1e-6 Å^-2 or user convention)
        self.thickness_bins = np.linspace(0.0, 200.0, 51) if thickness_bins is None else np.asarray(thickness_bins)
        self.roughness_bins = np.linspace(0.0, 10.0, 21) if roughness_bins is None else np.asarray(roughness_bins)
        self.sld_bins = np.linspace(1.0, 6.0, 26) if sld_bins is None else np.asarray(sld_bins)

    # --------------------------
    # Persistence
    # --------------------------
    def state_dict(self):
        return {
            "thickness_bins": self.thickness_bins.astype(float).tolist(),
            "roughness_bins": self.roughness_bins.astype(float).tolist(),
            "sld_bins": self.sld_bins.astype(float).tolist(),
        }

    def load_state_dict(self, state: dict):
        self.thickness_bins = np.asarray(state["thickness_bins"], dtype=float)
        self.roughness_bins = np.asarray(state["roughness_bins"], dtype=float)
        self.sld_bins = np.asarray(state["sld_bins"], dtype=float)

    # --------------------------
    # Fitting helper
    # --------------------------
    def fit_from_dataset(self, h5_path: str, sample_n: int = 1000, rng_seed: int = 0):
        """Estimate reasonable bins from real dataset values (sample subset)."""
        rng = np.random.default_rng(rng_seed)
        with h5py.File(h5_path, "r") as hf:
            n = hf["R"].shape[0]
            idx = rng.choice(n, size=min(sample_n, n), replace=False)
            thickness = hf["thickness"][idx]  # shape (k, L)
            roughness = hf["roughness"][idx]
            sld = hf["sld"][idx]

        # flatten and compute percentiles
        tvals = np.asarray(thickness).ravel()
        rvals = np.asarray(roughness).ravel()
        svals = np.asarray(sld).ravel()

        # avoid NaNs
        tvals = tvals[np.isfinite(tvals)]
        rvals = rvals[np.isfinite(rvals)]
        svals = svals[np.isfinite(svals)]

        # set bins using logspace for thickness if range large
        tmin, tmax = max(tvals.min(), 1e-6), tvals.max()
        if tmax / max(tmin, 1e-12) > 50:
            # use logspace for thickness
            self.thickness_bins = np.logspace(np.log10(tmin), np.log10(tmax), 51)
        else:
            self.thickness_bins = np.linspace(tmin, tmax, 51)

        self.roughness_bins = np.linspace(max(0.0, rvals.min()), rvals.max(), 21)
        self.sld_bins = np.linspace(svals.min(), svals.max(), 26)

    # --------------------------
    # Quantize API
    # --------------------------
    @staticmethod
    def _quantize_array(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Vectorized quantization. Returns indices clipped to valid range [0, len(bins)-2]."""
        idx = np.digitize(vals, bins) - 1
        idx = np.clip(idx, 0, len(bins) - 2)
        return idx.astype(np.int64)

    def quantize(self, param: ParamSet) -> np.ndarray:
        """Accept ParamSet (single) or array-like -> returns np.ndarray labels."""
        t = float(param.thickness)
        r = float(param.roughness)
        s = float(param.sld)
        return np.array([
            self._quantize_array(np.asarray([t]), self.thickness_bins)[0],
            self._quantize_array(np.asarray([r]), self.roughness_bins)[0],
            self._quantize_array(np.asarray([s]), self.sld_bins)[0],
        ], dtype=np.int64)

    def quantize_multi(self, thickness_arr: np.ndarray, roughness_arr: np.ndarray, sld_arr: np.ndarray) -> np.ndarray:
        """Vectorized quantize for arrays per-layer. Returns shape (n_layers, 3)."""
        t_idx = self._quantize_array(np.asarray(thickness_arr).ravel(), self.thickness_bins)
        r_idx = self._quantize_array(np.asarray(roughness_arr).ravel(), self.roughness_bins)
        s_idx = self._quantize_array(np.asarray(sld_arr).ravel(), self.sld_bins)
        return np.stack([t_idx, r_idx, s_idx], axis=1)


class DatasetH5(Dataset):
    """
    HDF5 dataset for XRR reflectivity and layer parameters.

    Key features:
      - per-process lazy HDF5 file opening (safe with num_workers>0)
      - optional log scaling and normalization (mean/std computed or provided)
      - returns (refl_tensor, labels_tensor) with CPU tensors (float32 / int64)
    """

    def __init__(
        self,
        h5_path: str,
        quantizer: ParamQuantizer,
        log_scale: bool = True,
        normalize: bool = True,
        downsample: int | None = None,
        stats: dict | None = None,
        prefetch_sample_n: int = 1000,
    ):
        self.h5_path = h5_path
        self.quantizer = quantizer
        self.log_scale = bool(log_scale)
        self.normalize = bool(normalize)
        self.downsample = None if downsample is None else int(downsample)
        self._hf = None  # per-process handle lazy opened
        self._length = None

        # open briefly to get length
        with h5py.File(self.h5_path, "r") as hf:
            self._length = hf["R"].shape[0]

        # compute stats if requested (mean/std on log(R))
        if stats is not None:
            self.mean = float(stats.get("mean", 0.0))
            self.std = float(stats.get("std", 1.0))
        elif self.normalize:
            # sample subset to estimate normalization
            self.mean, self.std = self._estimate_stats(prefetch_sample_n)
        else:
            self.mean, self.std = 0.0, 1.0

    def __len__(self) -> int:
        return int(self._length)

    # --------------------------
    # HDF5 access helpers
    # --------------------------
    def _ensure_open(self) -> h5py.File:
        """Open file per-process (safe with num_workers spawn/fork)."""
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")

    def _estimate_stats(self, n_sample: int = 1000) -> tuple[float, float]:
        """Estimate mean/std of log10(R) using random subset."""
        rng = np.random.default_rng(0)
        with h5py.File(self.h5_path, "r") as hf:
            n = hf["R"].shape[0]
            idx = rng.choice(n, size=min(n_sample, n), replace=False)
            vals = []
            for i in idx:
                R = hf["R"][i]
                if self.log_scale:
                    # guard against zeros/negatives
                    R = np.asarray(R, dtype=float)
                    R = np.where(R <= 0, 1e-15, R)
                    vals.append(np.log10(R))
                else:
                    vals.append(np.asarray(R, dtype=float))
            arr = np.concatenate([v.ravel() for v in vals], axis=0)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0, 1.0
        return float(arr.mean()), float(arr.std() if arr.std() > 0 else 1.0)

    # --------------------------
    # Data conversion
    # --------------------------
    def _process_reflectivity(self, refl: np.ndarray) -> np.ndarray:
        """Apply log-scaling, downsampling, normalization and return float32 array."""
        arr = np.asarray(refl, dtype=float).ravel()
        if self.downsample is not None and self.downsample > 0:
            # simple uniform downsampling
            if arr.size > self.downsample:
                idx = np.linspace(0, arr.size - 1, num=self.downsample, dtype=int)
                arr = arr[idx]
        if self.log_scale:
            arr = np.where(arr <= 0, 1e-15, arr)
            arr = np.log10(arr)
        if self.normalize:
            arr = (arr - self.mean) / (self.std if self.std != 0 else 1.0)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        self._ensure_open()
        hf = self._hf

        refl = hf["R"][idx]           # (q_len,) or (q_len, ...)
        thick = hf["thickness"][idx]  # (n_layers,)
        rough = hf["roughness"][idx]
        sld = hf["sld"][idx]

        # process reflectivity -> cpu numpy
        refl_arr = self._process_reflectivity(refl)

        # quantize labels vectorized
        labels = self.quantizer.quantize_multi(np.asarray(thick), np.asarray(rough), np.asarray(sld))
        # labels shape (n_layers, 3) as int64

        # convert to torch tensors (CPU). Device transfer is left to DataLoader / training loop.
        refl_t = torch.from_numpy(refl_arr).to(dtype=torch.float32)
        label_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        return refl_t, label_t

    # optional convenience
    def get_normalization(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std)}
