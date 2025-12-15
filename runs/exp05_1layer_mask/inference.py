import json

from pathlib import Path
import numpy as np
import torch

from config import CONFIG
from dataset import XRRPreprocessor
from xrr_model import XRR1DRegressor
from reflecto.math_utils import powerspace  # [필수] powerspace 임포트

class XRRInferenceEngine:
    def __init__(self, exp_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Device: {self.device}")

        # 1. 경로 설정
        if exp_dir is None:
            exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
        else:
            exp_dir = Path(exp_dir)

        self.stats_file = exp_dir / "stats.pt"
        self.checkpoint_file = exp_dir / "best.pt"
        self.config_file = exp_dir / "config.json"

        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                self.train_config = json.load(f)
            print(f"[Inference] Loaded training config from {self.config_file}")
        else:
            print("[Inference] Warning: config.json not found. Fallback to current global CONFIG.")
            self.train_config = CONFIG

        # 2. Master Grid 설정 (학습 때와 동일해야 함!)
        sim_conf = self.train_config["simulation"]
        q_min = sim_conf["q_min"]
        q_max = sim_conf["q_max"]
        n_points = sim_conf["q_points"]
        power = sim_conf["power"]

        print(f"[Inference] Grid Generation: Power={power}, Points={n_points}")

        # 모델이 바라보는 세상(Grid)을 학습 때와 똑같이 만듭니다.
        self.target_qs = powerspace(q_min, q_max, n_points, power=power).astype(np.float32)

        # 3. 전처리기 초기화
        # 여기서 target_qs를 넘겨주면, process_input에서
        # 들어오는 모든 데이터(등간격)를 이 target_qs(비선형)에 맞춰 내삽합니다.
        self.processor = XRRPreprocessor(
            qs=self.target_qs,
            stats_file=self.stats_file,
            device=self.device
        )

        # 4. 모델 로드
        self._load_model()

    def _load_model(self):
        ckpt = torch.load(self.checkpoint_file, map_location=self.device)

        # Config Fallback
        model_args = ckpt.get('config', {}).get('model_args', {
            'q_len': len(self.target_qs),
            'input_channels': 2,
            'n_channels': CONFIG["model"]["n_channels"],
            'depth': CONFIG["model"]["depth"],
            'mlp_hidden': CONFIG["model"]["mlp_hidden"],
        })

        self.model = XRR1DRegressor(**model_args).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"[Inference] Model loaded from {self.checkpoint_file}")

    def predict(self, q_raw, R_raw):
        """
        q_raw: 실측 데이터의 q (등간격이어도 상관없음)
        R_raw: 실측 데이터의 R
        """
        # 1. 전처리 (Resampling)
        # 내부적으로 np.interp(self.target_qs, q_raw, R_raw)가 실행됩니다.
        # 즉, 등간격 데이터를 모델이 아는 Powerspace 간격으로 변환합니다.
        x = self.processor.process_input(q_raw, R_raw).unsqueeze(0).to(self.device)

        # 2. 추론
        with torch.no_grad():
            y_pred_norm = self.model(x).squeeze(0)

        # 3. 역정규화
        y_pred = self.processor.denormalize_params(y_pred_norm)

        return y_pred
