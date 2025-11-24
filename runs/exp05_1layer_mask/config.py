import json
from pathlib import Path

import numpy as np

from reflecto.physics_utils import tth2q

CONFIG = {
    "exp_name": "power1_8",
    # "base_dir": Path(r"D:\Members\IsaacYong\Data\XRR_AI\one_layer"),
    "base_dir": Path(r"D:\data\XRR_AI\mask"),
    "param_ranges": {
        "thickness": None,
        "roughness": None,
        "sld": None,
        "sio2_thickness": None,
        "sio2_roughness": None,
        "sio2_sld": None,
    },
    "simulation": {
        "wavelength": 1.54,
        "n_samples": int(1e6),
        "q_points": 1000,
        "q_min": tth2q(0.1),
        "q_max": tth2q(15),
        "power": 1.8
    },
    "model": {
        "n_channels": 64,
        "depth": 4,
        "mlp_hidden": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 64,
        "epochs": 50,
        "lr": 0.002,
        "weight_decay": 1e-4,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "patience" : 15,
        "num_workers": 0
    },
}

def save_config(config: dict, file: Path | str):
    """
    딕셔너리를 JSON 파일로 저장합니다.
    - Path -> str 변환
    - Numpy 타입 -> python native 타입 변환
    - 재귀적 탐색
    - 듣도 보도 못한 타입(Custom Class 등) -> repr() 문자열로 변환하여 저장 (Fallback)
    """

    def _to_serializable(obj):
        # 1. 딕셔너리 재귀 호출
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}

        # 2. 리스트/튜플 재귀 호출
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]

        # 3. 이미 JSON 처리가 가능한 기본 타입들은 그대로 반환 (중요!)
        # 이걸 안 하면 str도 repr() 타서 "'문자열'" 처럼 따옴표가 이중으로 저장됨
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # 4. 특수 타입 처리
        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # 5. [요청하신 기능] 그 외 모든 알 수 없는 타입은 repr() 문자열로 저장
        try:
            return repr(obj)
        except Exception:
            return "<Unserializable Object>"

    # 변환 실행
    safe_config = _to_serializable(config)

    # 경로 생성 및 저장
    save_path = Path(file)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(safe_config, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    from pprint import pprint

    pprint(CONFIG)
