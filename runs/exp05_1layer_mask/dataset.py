from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class XRRPreprocessor:
    """
    Shared class for XRR data preprocessing and inverse transformation.
    (Used by both Dataset and InferenceEngine)
    """
    def __init__(self,
        qs: np.ndarray,
        stats_file: Path | str | None = None,
        device: torch.device = torch.device('cpu')
    ):
        # 1. Set up Master Grid
        self.target_q = qs
        self.device = device
        self.param_mean = None
        self.param_std = None

        if stats_file and Path(stats_file).exists():
            self.load_stats(stats_file)

    def load_stats(self, stats_file):
        """Load statistics file"""
        stats = torch.load(stats_file, map_location=self.device)
        self.param_mean = stats["param_mean"]
        self.param_std = stats["param_std"]

    def process_input(self, q_raw, R_raw):
        """
        Raw Data (q, R) -> Model Input Tensor (2, N)
        Operation: Normalization + Interpolation + Masking
        """
        R_max = np.max(R_raw)
        R_norm = R_raw / (R_max + 1e-15)
        R_log = np.log10(np.maximum(R_norm, 1e-15))

        # Sort (Reverse if q is in descending order)
        if q_raw[0] > q_raw[-1]:
            q_raw = q_raw[::-1]
            R_log = R_log[::-1]

        R_interp = np.interp(self.target_q, q_raw, R_log, left=0.0, right=0.0)
        q_valid_mask = (self.target_q >= np.min(q_raw)) & (self.target_q <= np.max(q_raw))

        R_tensor = torch.from_numpy(R_interp.astype(np.float32))
        mask_tensor = torch.from_numpy(q_valid_mask.astype(np.float32))

        return torch.stack([R_tensor, mask_tensor], dim=0)

    def denormalize_params(self, params_norm):
        """
        Model Output (Norm) -> Physical Values
        """
        if self.param_mean is None:
            raise ValueError("Statistics file is not loaded.")
        if isinstance(params_norm, torch.Tensor):
            params_norm = params_norm.detach().cpu().numpy()

        # CPU Numpy Operation (Convert to Numpy if mean/std are Tensors)
        # Assumes mean/std are loaded as Tensors in load_stats
        mean = self.param_mean.cpu().numpy() if isinstance(self.param_mean, torch.Tensor) else self.param_mean
        std = self.param_std.cpu().numpy() if isinstance(self.param_std, torch.Tensor) else self.param_std

        params_real = params_norm * std + mean
        params_real = np.maximum(params_real, 1e-3)

        return params_real

    def normalize_parameters(self, params_real):
        """Physical Values -> Model Target (Norm)"""
        mean = self.param_mean.numpy() if isinstance(self.param_mean, torch.Tensor) else self.param_mean
        std = self.param_std.numpy() if isinstance(self.param_std, torch.Tensor) else self.param_std

        params_norm = (params_real - mean) / std
        return torch.from_numpy(params_norm.astype(np.float32))


class XRR1LayerDataset(Dataset):
    """
    XRR 1-layer Dataset with:
      1. Global Grid Alignment (Interpolation)
      2. Dynamic Masking (For variable scan ranges)
      3. Realistic Random Crop Augmentation (Protecting Critical Angle)
      4. Robust Normalization
    """

    def __init__(
        self, qs: np.ndarray, h5_file: str | Path, stats_file: str | Path,
        mode: str = "train", val_ratio: float = 0.2, test_ratio: float = 0.1,
        augment: bool = False, aug_prob: float = 0.5, min_scan_range: float = 0.15
    ):

        self.h5_path = Path(h5_file)
        self.stats_path = Path(stats_file)
        self.mode = mode

        # Grid & Augmentation
        self.target_q = qs
        self.augment = augment and (mode == 'train')
        self.aug_prob = aug_prob
        self.min_scan_range = min_scan_range

        self.hf: h5py.File | None = None
        # Load Data
        self._load_h5_data()

        # Data Split
        self._setup_split(val_ratio, test_ratio)

        # Setup Normalization Statistics
        self.processor = XRRPreprocessor(self.target_q)
        self._setup_param_stats()
        self.processor.load_stats(self.stats_path)

    def _load_h5_data(self):
        """Load entire data from H5 file into memory"""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            # q data: (N, L) or (L,)
            self.source_q = hf["q"][:]

            # Parameters and Reflectivity
            self.thickness = hf["thickness"][:].squeeze()
            self.roughness = hf["roughness"][:].squeeze()
            self.sld = hf["sld"][:].squeeze()

            self.n_total = hf["R"].shape[0]

    def _setup_split(self, val_ratio, test_ratio):
        """Split indices for Train/Val/Test"""
        train_ratio = 1.0 - val_ratio - test_ratio
        self.train_end = int(self.n_total * train_ratio)
        self.val_end = int(self.n_total * (train_ratio + val_ratio))

        match self.mode:
            case "train":
                self.indices = range(0, self.train_end)
            case "val":
                self.indices = range(self.train_end, self.val_end)
            case "test":
                self.indices = range(self.val_end, self.n_total)
            case _:
                raise ValueError(f"Unknown mode: {self.mode}")

    def _setup_param_stats(self):
        """Process parameter normalization statistics (Calculate only in Train mode)"""
        if self.stats_path.exists():
            print(f"[{self.mode}] Loading statistics from {self.stats_path}")
            stats = torch.load(self.stats_path)
            self.param_mean = stats["param_mean"].numpy()
            self.param_std = stats["param_std"].numpy()

        elif self.mode == "train":
            print(f"[{self.mode}] Calculating statistics from training data...")
            train_indices = range(0, self.train_end)
            params = np.stack([
                self.thickness[train_indices],
                self.roughness[train_indices],
                self.sld[train_indices],
            ], axis=1)

            self.param_mean = np.mean(params, axis=0)
            self.param_std = np.std(params, axis=0) + 1e-8 # Prevent division by zero

            torch.save({
                "param_mean": torch.from_numpy(self.param_mean),
                "param_std": torch.from_numpy(self.param_std)
            }, self.stats_path)
        else:
            raise FileNotFoundError(f"Stats file not found at {self.stats_path}. Run 'train' first.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        R_raw, q_raw, params_raw = self._get_raw_data(real_idx)

        # Augmentation (Dataset's unique role)
        if self.augment:
            R_raw, q_raw = self._apply_augmentation(R_raw, q_raw)

        # [NEW] Delegate complex processing to processor!
        input_tensor = self.processor.process_input(q_raw, R_raw)
        params_tensor = self.processor.normalize_parameters(params_raw)

        return input_tensor, params_tensor

    def _get_raw_data(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r', swmr=True)

        R_raw = self.hf["R"][idx]

        if self.source_q.ndim == 1:
            q_raw = self.source_q
        else:
            q_raw = self.source_q[idx]

        params_raw = np.array([
            self.thickness[idx],
            self.roughness[idx],
            self.sld[idx],
        ], dtype=np.float32)

        return R_raw, q_raw, params_raw

    def _apply_augmentation(self, R_raw, q_raw):
        """
        [Realistic Augmentation]
        - Front (Beamstop): Randomly crop up to max 0.04 (Protect critical angle)
        - Back (Signal Loss): Boldly crop within remaining margin
        """
        if np.random.rand() > self.aug_prob:
            return R_raw, q_raw

        current_min = q_raw[0]
        current_max = q_raw[-1]

        # Total Slack available for cropping
        slack = (current_max - current_min) - self.min_scan_range

        if slack <= 0:
            return R_raw, q_raw

        # 1. Front Crop: Limit to max 0.04 to protect critical angle (0.03~0.04)
        # Crop up to the smaller of 20% of slack or 0.04
        max_front_limit = 0.04
        front_crop_limit = min(slack * 0.2, max_front_limit)

        crop_start = np.random.uniform(0, front_crop_limit)

        # 2. Back Crop: Allocate remaining slack to back cropping
        remaining_slack = slack - crop_start
        crop_end = np.random.uniform(0, remaining_slack)

        # 3. Slicing
        new_min = current_min + crop_start
        new_max = current_max - crop_end

        mask = (q_raw >= new_min) & (q_raw <= new_max)

        # Safety: Return original if too few data points remain
        if np.sum(mask) < 10:
            return R_raw, q_raw

        return R_raw[mask], q_raw[mask]
    def __del__(self):
        if self.hf is not None:
            self.hf.close()
