from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from augmentations import XRRAugmentations


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
        Raw Data (Linear q) -> Model Input Tensor (Power q)
        """
        # 1. 데이터 정제 (NaN 방지)

        R_raw = np.nan_to_num(R_raw, nan=1e-15, posinf=1e-15, neginf=1e-15)

        # 2. Normalize (Max=1.0) -> Log Scale
        R_max = np.max(R_raw)
        if R_max <= 0:
            R_max = 1.0 # 0 나누기 방지

        R_norm = R_raw / R_max
        R_log = np.log10(np.maximum(R_norm, 1e-15))

        # 3. 오름차순 정렬 (np.interp는 x가 정렬되어 있어야 함)
        if q_raw[0] > q_raw[-1]:
            q_raw = q_raw[::-1]
            R_log = R_log[::-1]

        padding_val = -15.0

        R_interp = np.interp(self.target_q, q_raw, R_log, left=padding_val, right=padding_val)

        # 모델 그리드(target_q) 중 실측 데이터(q_raw) 범위 안에 있는 것만 유효(1)
        q_valid_mask = (self.target_q >= np.min(q_raw)) & (self.target_q <= np.max(q_raw))

        # 5. Tensor 변환
        R_tensor = torch.from_numpy(R_interp.astype(np.float32))
        mask_tensor = torch.from_numpy(q_valid_mask.astype(np.float32))

        # (2, N) 형태로 반환 [LogR, Mask]
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
        augment: bool = False, expand_factor: int = 1, aug_prob: float = 0.5,
        min_scan_range: float = 0.15, q_shift_sigma: float = 0.002, intensity_scale: float = 0.1
    ):

        self.h5_path = Path(h5_file)
        self.stats_path = Path(stats_file)
        self.mode = mode

        # Grid & Augmentation
        self.target_q = qs
        self.augment = augment and (mode == 'train')
        self.expand_factor = expand_factor if (mode == 'train') else 1
        self.min_scan_range = min_scan_range
        self.aug_prob = aug_prob
        if self.augment:
            self.physics_augmenter = XRRAugmentations(
                intensity_noise_scale=intensity_scale,
                q_shift_sigma=q_shift_sigma,
                prob=aug_prob
            )

        self.hf: h5py.File | None = None
        # Load Data
        self._load_metadata_only()

        # Data Split
        self._setup_split(val_ratio, test_ratio)

        # Setup Normalization Statistics
        self.processor = XRRPreprocessor(self.target_q)

        if mode == 'train' and not self.stats_path.exists():
            self._calculate_and_save_stats()

        # self._setup_param_stats()
        self.processor.load_stats(self.stats_path)

    def _load_metadata_only(self):
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            self.source_q = hf["q"][:]
            self.n_total = hf["R"].shape[0]

            # Load small parameter arrays into memory for speed
            self.thickness = hf["thickness"][:].squeeze()
            self.roughness = hf["roughness"][:].squeeze()
            self.sld = hf["sld"][:].squeeze()

    def _calculate_and_save_stats(self):
        """Calculate normalization stats on training subset."""
        print(f"[{self.mode}] Calculating statistics...")
        idx = range(self.train_end) # Use only training part

        params_list = [
            self.thickness[idx], self.roughness[idx], self.sld[idx]
        ]

        params = np.stack(params_list, axis=1)
        mean = np.mean(params, axis=0)
        std = np.std(params, axis=0) + 1e-8

        torch.save({
            "param_mean": torch.from_numpy(mean),
            "param_std": torch.from_numpy(std)
        }, self.stats_path)

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

    def __len__(self):

        return len(self.indices) * self.expand_factor

    def __getitem__(self, idx):
        # Map expanded index back to original range
        # e.g. If len=100, idx=105 -> real_idx=indices[5]
        original_idx = idx % len(self.indices)
        real_idx = self.indices[original_idx]

        R_raw, q_raw, params_raw = self._get_raw_data_lazy(real_idx)

        if self.augment:
            R_raw, q_raw = self._apply_crop_augmentation(R_raw, q_raw)
            q_raw, R_raw = self.physics_augmenter(q_raw, R_raw)

        # Preprocessing (Log -> Interp -> Mask -> Normalize)
        input_tensor = self.processor.process_input(q_raw, R_raw)
        params_tensor = self.processor.normalize_parameters(params_raw)

        return input_tensor, params_tensor

    def _get_raw_data_lazy(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r', swmr=True)

        R_raw = self.hf["R"][idx]
        q_raw = self.source_q[idx] if self.source_q.ndim == 2 else self.source_q

        p_list = [self.thickness[idx], self.roughness[idx], self.sld[idx]]

        return R_raw, q_raw, np.array(p_list, dtype=np.float32)

    def _apply_crop_augmentation(self, R_raw, q_raw):
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
        if self.hf:
            self.hf.close()
