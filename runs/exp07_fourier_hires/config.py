import json
from pathlib import Path

import numpy as np

from reflecto_exp.physics_utils import tth2q

CONFIG = {
    "exp_name": "sio2_uniform",
    "base_dir": Path(r"D:\Members\IsaacYong\Data\XRR_AI\exp07"),
    # "base_dir": Path(r"C:\Warehouse\data\XRR_AI\exp07"),

    "param_ranges": {
        "thickness": (10.0, 1200.0),
        "roughness": (0.0, 15.0),
        "sld": (5.0, 100.0),
        "sio2_thickness": (10.0, 30.0),
        "sio2_roughness": (1.0, 5.0),
        "sio2_sld": (5.0, 22.0),
    },
    "simulation": {
        "wavelength": 1.54,
        "n_samples": int(1e5),
        "q_points": 2000,
        "q_min": tth2q(0.05),
        "q_max": tth2q(15.0),
    },
    "model": {
        "use_fourier": True,
        "fourier_scale": 15.0,
        "n_channels": 64,
        "depth": 6,
        "mlp_hidden": 512,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 64,
        "epochs": 150,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "patience": 20,
        "num_workers": 4,

        "augment": True,
        "expand_factor": 20,
        "aug_prob": 0.9,
        "intensity_scale": 0.25,
        "q_shift_sigma": 0.004,
    },
}

def save_config(config: dict, file: Path | str):
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
