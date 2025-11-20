from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class XRR1LayerDataset(Dataset):
    """
    XRR 1-layer Dataset with:
      1. Global Grid Alignment (Interpolation)
      2. Dynamic Masking (For variable scan ranges)
      3. Realistic Random Crop Augmentation (Protecting Critical Angle)
      4. Robust Normalization
    """

    def __init__(self, h5_file: str | Path, stats_file: str | Path,
                 mode: str = "train", val_ratio: float = 0.2, test_ratio: float = 0.1,
                 q_min: float = 0.0, q_max: float = 0.5, n_points: int = 200,
                 augment: bool = False, aug_prob: float = 0.5, min_scan_range: float = 0.15):

        self.h5_path = Path(h5_file)
        self.stats_path = Path(stats_file)
        self.mode = mode

        # ---------------------------------------------------------
        # 1. Master Grid 설정 (모델이 보는 고정된 세상)
        # ---------------------------------------------------------
        self.target_q = np.linspace(q_min, q_max, n_points).astype(np.float32)
        self.n_points = n_points

        # ---------------------------------------------------------
        # 2. Augmentation 설정 (Train 모드일 때만 활성화)
        # ---------------------------------------------------------
        self.augment = augment and (mode == 'train')
        self.aug_prob = aug_prob
        self.min_scan_range = min_scan_range

        # 3. 데이터 로드
        self._load_h5_data()

        # 4. 데이터 분할 (Split)
        self._setup_split(val_ratio, test_ratio)

        # 5. 정규화 통계 설정
        self._setup_param_stats()

    def _load_h5_data(self):
        """H5 파일에서 전체 데이터를 메모리에 로드"""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            # q 데이터: (N, L) 또는 (L,)
            self.source_q = hf["q"][:]

            # 파라미터 및 반사율
            self.thickness = hf["thickness"][:].squeeze()
            self.roughness = hf["roughness"][:].squeeze()
            self.sld = hf["sld"][:].squeeze()
            self.reflectivity = hf["R"][:]

        self.n_total = len(self.reflectivity)

    def _setup_split(self, val_ratio, test_ratio):
        """Train/Val/Test 인덱스 분할"""
        train_ratio = 1.0 - val_ratio - test_ratio
        self.train_end = int(self.n_total * train_ratio)
        self.val_end = int(self.n_total * (train_ratio + val_ratio))

        if self.mode == "train":
            self.indices = range(0, self.train_end)
        elif self.mode == "val":
            self.indices = range(self.train_end, self.val_end)
        elif self.mode == "test":
            self.indices = range(self.val_end, self.n_total)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _setup_param_stats(self):
        """파라미터 정규화 통계 처리 (Train에서만 계산)"""
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
                self.sld[train_indices]
            ], axis=1)

            self.param_mean = np.mean(params, axis=0)
            self.param_std = np.std(params, axis=0) + 1e-8 # 0 나누기 방지

            torch.save({
                "param_mean": torch.from_numpy(self.param_mean),
                "param_std": torch.from_numpy(self.param_std)
            }, self.stats_path)
        else:
            raise FileNotFoundError(f"Stats file not found at {self.stats_path}. Run 'train' first.")

    def __len__(self):
        return len(self.indices)

    # =========================================================================
    #  Data Processing Pipeline
    # =========================================================================
    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # 1. Raw Data 추출
        R_raw, q_raw, params_raw = self._get_raw_data(real_idx)

        # 2. 현실적인 Augmentation (Train Only)
        if self.augment:
            R_raw, q_raw = self._apply_augmentation(R_raw, q_raw)

        # 3. Reflectivity 정규화 (Log Scale)
        R_log = self._normalize_reflectivity(R_raw)

        # 4. Grid 정렬 및 마스킹 (2 Channel Tensor 생성)
        input_tensor = self._align_and_mask(R_log, q_raw)

        # 5. Parameters 정규화
        params_tensor = self._normalize_parameters(params_raw)

        return input_tensor, params_tensor

    # =========================================================================
    #  Functional Components
    # =========================================================================
    def _get_raw_data(self, idx):
        R_raw = self.reflectivity[idx]

        if self.source_q.ndim == 2:
            q_raw = self.source_q[idx]
        else:
            q_raw = self.source_q

        params_raw = np.array([
            self.thickness[idx],
            self.roughness[idx],
            self.sld[idx]
        ], dtype=np.float32)

        return R_raw, q_raw, params_raw

    def _apply_augmentation(self, R_raw, q_raw):
        """
        [Realistic Augmentation]
        - Front (Beamstop): 최대 0.04까지만 랜덤하게 자름 (임계각 보호)
        - Back (Signal Loss): 남은 여유분 내에서 과감하게 자름
        """
        if np.random.rand() > self.aug_prob:
            return R_raw, q_raw

        current_min = q_raw[0]
        current_max = q_raw[-1]

        # 자를 수 있는 총 여유분 (Total Slack)
        slack = (current_max - current_min) - self.min_scan_range

        if slack <= 0:
            return R_raw, q_raw

        # 1. Front Crop: 임계각(0.03~0.04) 보호를 위해 최대 0.04로 제한
        # slack의 20%와 0.04 중 작은 값까지만 자름
        max_front_limit = 0.04
        front_crop_limit = min(slack * 0.2, max_front_limit)

        crop_start = np.random.uniform(0, front_crop_limit)

        # 2. Back Crop: 남은 여유분(Slack)을 뒤쪽 자르는데 할당
        remaining_slack = slack - crop_start
        crop_end = np.random.uniform(0, remaining_slack)

        # 3. Slicing
        new_min = current_min + crop_start
        new_max = current_max - crop_end

        mask = (q_raw >= new_min) & (q_raw <= new_max)

        # 안전장치: 데이터 포인트가 너무 적으면 원본 반환
        if np.sum(mask) < 10:
            return R_raw, q_raw

        return R_raw[mask], q_raw[mask]

    def _normalize_reflectivity(self, R_raw):
        """Max Norm -> Log10"""
        R_max = np.max(R_raw)
        R_norm = R_raw / (R_max + 1e-15)
        R_log = np.log10(np.maximum(R_norm, 1e-15))
        return R_log

    def _align_and_mask(self, R_log, q_raw):
        """Master Grid로 내삽 및 Mask 생성 -> [2, N] Tensor"""
        # 1. Interpolation (데이터가 없는 곳은 0.0으로 Padding)
        # 주의: Log Scale에서 0.0은 실제로는 1.0(전반사)을 의미할 수도, 혹은 -15(매우 작은 값)일 수도 있음.
        # 하지만 여기서는 Mask 채널이 0이 되므로, 이 값이 무엇이든 모델은 무시함.
        R_interp = np.interp(self.target_q, q_raw, R_log, left=0.0, right=0.0)

        # 2. Mask Generation (데이터가 존재하는 q 범위만 1.0)
        q_valid_mask = (self.target_q >= np.min(q_raw)) & (self.target_q <= np.max(q_raw))

        # 3. Convert to Tensor & Stack
        R_tensor = torch.from_numpy(R_interp.astype(np.float32))
        mask_tensor = torch.from_numpy(q_valid_mask.astype(np.float32))

        # Shape: (2, n_points) -> [LogR, Mask]
        return torch.stack([R_tensor, mask_tensor], dim=0)

    def _normalize_parameters(self, params_raw):
        """Z-score Normalization"""
        params_norm = (params_raw - self.param_mean) / self.param_std
        return torch.from_numpy(params_norm).float()

    # =========================================================================
    #  Utils (Denormalization)
    # =========================================================================
    def denormalize_params(self, params_norm):
        if isinstance(params_norm, torch.Tensor):
            params_norm = params_norm.detach().cpu().numpy()
        return params_norm * self.param_std + self.param_mean

    @staticmethod
    def denormalize_reflectivity(R_log_norm, original_max):
        if isinstance(R_log_norm, torch.Tensor):
            R_log_norm = R_log_norm.detach().cpu().numpy()
        return (10 ** R_log_norm) * original_max
