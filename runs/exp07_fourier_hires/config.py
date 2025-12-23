import json
from pathlib import Path

import numpy as np

from reflecto_exp.physics_utils import tth2q

CONFIG = {
    "exp_name": "test",
    # "base_dir": Path(r"D:\Members\IsaacYong\Data\XRR_AI\exp07"),
    "base_dir": Path(r"C:\Warehouse\data\XRR_AI\exp07"),

    "param_ranges": {
        "thickness": (10.0, 1200.0),  # 1200A까지 커버 (고주파수 극한 테스트)
        "roughness": (0.0, 40.0),     # 매우 거친 표면까지
        "sld": (5.0, 100.0),          # 금속/세라믹 등 다양한 물질
        "sio2_thickness": (10.0, 30.0),
        "sio2_roughness": (1.0, 5.0),
        "sio2_sld": (5.0, 22.0),
    },
    "simulation": {
        "wavelength": 1.54,
        "n_samples": int(5e4),        # 기본 5만장 (Augment로 x50배 효과)
        "q_points": 2000,             # [수정] 500 -> 2000 (Nyquist 안전권 확보)
        "q_min": tth2q(0.05),         # 저각 정보 강화
        "q_max": tth2q(15.0),
    },
    "model": {
        "use_fourier": True,
        "fourier_scale": 15.0,        # [수정] 주파수 대역폭 증가
        "n_channels": 64,
        "depth": 6,
        "mlp_hidden": 512,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 64,
        "epochs": 100,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "patience": 20,
        "num_workers": 4,

        # [Point 3] 학습 볼륨 증대 (Online Augmentation)
        "augment": True,
        "expand_factor": 50,          # Epoch당 데이터셋을 50번 재사용 (매번 다른 노이즈)
        "aug_prob": 0.9,              # 90% 확률로 증강
        "intensity_scale": 0.25,      # I0 스케일링 오차 +/- 25%
        "q_shift_sigma": 0.004,       # q축 틀어짐 강화
    },
}

def save_config(config: dict, file: Path | str):
    # (기존 io_utils/config 코드와 동일하게 JSON 저장 구현)
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, str | int | float | bool | type(None)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer | np.floating):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return repr(obj)

    save_path = Path(file)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(config), f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    from pprint import pprint
    pprint(CONFIG)
