from pathlib import Path

# In2O3 30nm one layer
_mean_thickness = 300
_mean_sld = 4.46
_thickness_range = (_mean_thickness * 0.8, _mean_thickness * 1.2)
_sld_range = (_mean_sld * 0.8, _mean_sld * 1.2)

# ==================== 설정 (하나의 파일에서 모두 관리) ====================
CONFIG = {
    "exp_name": "genx_In2O3_simple",
    "base_dir": Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer"),
    "param_ranges": {
        "thickness": _thickness_range,
        "roughness": (0.0, 10.0),
        "sld": _sld_range,
    },
    "simulation": {
        "n_samples": 1_000_000,
        "q_points": 200,
        "wavelength": 1.54,
        "tth_min": 0.1,
        "tth_max": 6.0,
    },
    "model": {
        "n_channels": 64,
        "depth": 4,
        "mlp_hidden": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "val_ratio": 0.2,
    },
}
