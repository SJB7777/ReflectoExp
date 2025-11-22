from pathlib import Path

from reflecto.physics_utils import tth2q

CONFIG = {
    "exp_name": "test_2",
    # "base_dir": Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer"),
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
        "n_samples": int(1e3),
        "q_points": 250,
        "wavelength": 1.54,
        "q_min": tth2q(0.1),
        "q_max": tth2q(15),
    },
    "model": {
        "n_channels": 64,
        "depth": 4,
        "mlp_hidden": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.002,
        "weight_decay": 1e-4,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "patience" : 15,
        "num_workers": 0
    },
}


if __name__ == "__main__":
    from pprint import pprint

    pprint(CONFIG)
