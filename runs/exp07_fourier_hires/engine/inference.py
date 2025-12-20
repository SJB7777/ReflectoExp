import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from xrr_model import XRRPhysicsModel

from reflecto_exp.math_utils import i0_normalize
from reflecto_exp.simulate.simul_genx import ParamSet, param2refl


class XRRPreprocessor:
    """XRR 데이터 전처리 및 역변환을 담당하는 유틸리티 클래스"""
    def __init__(self, qs: np.ndarray, stats_file: Path | str | None = None, device: torch.device = torch.device('cpu')):
        self.target_q = qs
        self.device = device
        self.param_mean = None
        self.param_std = None

        if stats_file and Path(stats_file).exists():
            self.load_stats(stats_file)

    def load_stats(self, stats_file: Union[Path, str]):
        stats = torch.load(stats_file, map_location=self.device, weights_only=True)
        self.param_mean = stats["param_mean"]
        self.param_std = stats["param_std"]

    def process_input(self, q_raw: np.ndarray, R_raw: np.ndarray) -> torch.Tensor:
        """실측 데이터(Linear)를 모델 입력용 텐서(Grid-aligned Log)로 변환"""
        R_raw = np.nan_to_num(R_raw, nan=1e-15, posinf=1e-15, neginf=1e-15)
        R_norm = i0_normalize(R_raw)
        R_log = np.log10(np.maximum(R_norm, 1e-15))

        # 정렬 보장
        if q_raw[0] > q_raw[-1]:
            q_raw, R_log = q_raw[::-1], R_log[::-1]

        # Resampling
        R_interp = np.interp(self.target_q, q_raw, R_log, left=-15.0, right=-15.0)
        q_valid_mask = (self.target_q >= np.min(q_raw)) & (self.target_q <= np.max(q_raw))

        return torch.stack([
            torch.from_numpy(R_interp.astype(np.float32)),
            torch.from_numpy(q_valid_mask.astype(np.float32))
        ], dim=0)

    def denormalize(self, params_norm: torch.Tensor) -> np.ndarray:
        """모델의 정규화된 출력을 실제 물리량(A, x10^-6 A^-2)으로 변환"""
        if self.param_mean is None:
            raise ValueError("Normalization statistics not loaded.")
        
        mean = self.param_mean.cpu().numpy() if torch.is_tensor(self.param_mean) else self.param_mean
        std = self.param_std.cpu().numpy() if torch.is_tensor(self.param_std) else self.param_std
        
        return params_norm.detach().cpu().numpy() * std + mean


class XRRInferenceEngine:
    def __init__(self, exp_dir: Union[str, Path], device: Optional[str] = None):
        self.exp_dir = Path(exp_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self._validate_paths()
        self._load_config()
        self._init_processor()
        self._load_model()

    def _validate_paths(self):
        required_files = ["config.json", "stats.pt", "best.pt"]
        for f in required_files:
            if not (self.exp_dir / f).exists():
                raise FileNotFoundError(f"Missing required file in {self.exp_dir}: {f}")

    def _load_config(self):
        with open(self.exp_dir / "config.json", "r") as f:
            self.config = json.load(f)

    def _init_processor(self):
        sim = self.config["simulation"]
        self.target_qs = np.linspace(sim["q_min"], sim["q_max"], sim["q_points"]).astype(np.float32)
        self.processor = XRRPreprocessor(self.target_qs, self.exp_dir / "stats.pt", self.device)

    def _load_model(self):
        m_conf = self.config["model"]
        model_args = {
            'q_len': len(self.target_qs),
            'input_channels': 2,
            'n_channels': m_conf["n_channels"],
            'depth': m_conf["depth"],
            'mlp_hidden': m_conf["mlp_hidden"],
            'dropout': 0.0,
            'use_fourier': m_conf.get("use_fourier", True),
            'fourier_scale': m_conf.get("fourier_scale", 10.0)
        }
        self.model = XRRPhysicsModel(**model_args).to(self.device)
        ckpt = torch.load(self.exp_dir / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    def predict(self, q_raw: np.ndarray, R_raw: np.ndarray) -> ParamSet:
        """AI 추론을 수행하고 ParamSet 객체를 반환"""
        x = self.processor.process_input(q_raw, R_raw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pred_norm = self.model(x).squeeze(0)

        y_real = self.processor.denormalize(y_pred_norm)
        # 물리적 제약조건 적용 (음수 방지)
        y_real = np.maximum(y_real, [1.0, 0.0, 0.0])

        return ParamSet(thickness=float(y_real[0]), roughness=float(y_real[1]), sld=float(y_real[2]))

    def plot_ai_guess(self, q_measured: np.ndarray, R_measured: np.ndarray, ai_params: ParamSet, save_path: Optional[Path] = None):
        """AI의 초기 추측값과 실측 데이터를 비교하는 플롯 생성"""
        # AI 예측 파라미터로 곡선 생성
        R_ai = param2refl(q_measured, [ai_params])

        plt.figure(figsize=(10, 6))
        plt.plot(q_measured, i0_normalize(R_measured), 'ko', markersize=3, alpha=0.3, label='Measured Data')
        plt.plot(q_measured, R_ai, 'r--', lw=2, label=f'AI Guess (d={ai_params.thickness:.1f}Å)')

        plt.yscale('log')
        plt.xlabel(r'$q$ ($\AA^{-1}$)')
        plt.ylabel('Reflectivity')
        plt.title('Initial AI Prediction (Guess)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, which='both', alpha=0.2)

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
