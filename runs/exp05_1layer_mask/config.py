from pathlib import Path

from reflecto.simulate.simul_genx import tth2q_wavelen

# ==================== 설정 (하나의 파일에서 모두 관리) ====================
CONFIG = {
    "exp_name": "sio2_random2",
    "base_dir": Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer"),
    "param_ranges": {
        "thickness": None,
        "roughness": None,
        "sld": None,
    },
    "simulation": {
        "n_samples": int(1e6),
        "q_points": 250,
        "wavelength": 1.54,
        "q_min": tth2q_wavelen(0.1),
        "q_max": tth2q_wavelen(15),
    },
    "model": {
        "n_channels": 64,
        "depth": 4,
        "mlp_hidden": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 512,
        "epochs": 50,
        "lr": 0.002,
        "weight_decay": 1e-4,
        "val_ratio": 0.2,
    },
}


if __name__ == "__main__":
    from pprint import pprint

    pprint(CONFIG)
