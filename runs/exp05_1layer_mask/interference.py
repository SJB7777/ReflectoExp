from pathlib import Path

import numpy as np
import torch
from config import CONFIG
from xrr_model import XRR1DRegressor


class XRRInferenceEngine:
    def __init__(self, exp_dir=None):
        """
        추론 엔진 초기화: 설정 로드, 모델 로드, 통계 로드
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Device: {self.device}")

        # 1. 경로 설정
        if exp_dir is None:
            exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
        else:
            exp_dir = Path(exp_dir)

        self.stats_file = exp_dir / "stats.pt"
        self.checkpoint_file = exp_dir / "best.pt"

        # 2. Master Grid 설정 (학습과 동일해야 함)
        self.q_min = CONFIG["simulation"]["q_min"]
        self.q_max = CONFIG["simulation"]["q_max"]
        self.n_points = CONFIG["simulation"]["q_points"]
        # print(self.q_min, self.q_max, self.n_points)
        # 모델이 바라보는 고정된 q축 생성
        self.target_q = np.linspace(self.q_min, self.q_max, self.n_points).astype(np.float32)

        # 3. 정규화 통계 로드
        if not self.stats_file.exists():
            raise FileNotFoundError(f"통계 파일이 없습니다: {self.stats_file}")

        stats = torch.load(self.stats_file, map_location=self.device)
        self.param_mean = stats["param_mean"]
        self.param_std = stats["param_std"]

        # 4. 모델 로드
        self._load_model()

    def _load_model(self):
        """모델 생성 및 가중치 로드"""
        if not self.checkpoint_file.exists():
            raise FileNotFoundError(f"체크포인트가 없습니다: {self.checkpoint_file}")

        # 체크포인트에서 모델 설정 가져오기 (없으면 Config 사용)
        ckpt = torch.load(self.checkpoint_file, map_location=self.device)
        model_args = ckpt.get('config', {}).get('model_args', {
            'q_len': self.n_points,
            'input_channels': 2, # [중요] 2채널 고정
            'n_channels': CONFIG["model"]["n_channels"],
            'depth': CONFIG["model"]["depth"],
            'mlp_hidden': CONFIG["model"]["mlp_hidden"],
        })

        self.model = XRR1DRegressor(**model_args).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"[Inference] Model loaded from {self.checkpoint_file}")

    def preprocess(self, q_raw, R_raw):
        """
        Raw Data -> Model Input Tensor 변환
        (Dataset.__getitem__ 로직과 동일하게 구현)
        """
        # 1. Normalize R (Max Norm -> Log10)
        R_max = np.max(R_raw)
        R_norm = R_raw / (R_max + 1e-15)
        R_log = np.log10(np.maximum(R_norm, 1e-15))

        # 2. Interpolation (Master Grid로 내삽)
        # q_raw가 오름차순이어야 np.interp가 정확함
        if q_raw[0] > q_raw[-1]:
            q_raw = q_raw[::-1]
            R_log = R_log[::-1]

        R_interp = np.interp(self.target_q, q_raw, R_log, left=0.0, right=0.0)

        # 3. Masking
        q_valid_mask = (self.target_q >= np.min(q_raw)) & (self.target_q <= np.max(q_raw))

        # 4. Tensor 변환 & Stacking
        R_tensor = torch.from_numpy(R_interp.astype(np.float32))
        mask_tensor = torch.from_numpy(q_valid_mask.astype(np.float32))

        # Shape: (2, n_points)
        input_tensor = torch.stack([R_tensor, mask_tensor], dim=0)

        # Batch 차원 추가: (1, 2, n_points)
        return input_tensor.unsqueeze(0).to(self.device)

    def predict(self, q_raw, R_raw):
        """
        단일 데이터 추론 수행
        Returns: (thickness, roughness, sld) -> 실제 물리 단위
        """
        # 전처리
        x = self.preprocess(q_raw, R_raw)

        # 추론
        with torch.no_grad():
            # Output: (1, 3) -> Normalized Params
            y_pred_norm = self.model(x).squeeze(0)

        # 역정규화 (Denormalization)
        # Z-score 역변환: X = Z * Std + Mean
        y_pred = y_pred_norm * self.param_std + self.param_mean

        # Tensor -> Numpy / List 변환
        return y_pred.cpu().numpy()
