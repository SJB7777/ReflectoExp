import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from train_fused import FusedTrainer


def main():
    config = {
        "exp_name": "exp04_physics_fused",
        "h5_path": r"D:\03_Resources\Data\XRR_AI\data/xrr_data.h5",
        "checkpoint_dir": r"D:\03_Resources\Data\XRR_AI\model\exp001_reg_tanh",
        "norm_ranges": {
            "thickness": [20.0, 150.0],
            "roughness": [0.0, 10.0],
            "sld": [0.0, 140.0]
        },

        "epochs": 20,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "val_ratio": 0.2,
        "save_every": 5,
        "physics_weight": 0.5,
        "patience": 10,
        # 하드웨어
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mixed_precision": True,
        "num_workers": 4,
        "resume_from": None,  # "path/to/checkpoint.pt"
    }

    trainer = FusedTrainer(config)
    if config.get("resume_from"):
        trainer.load_checkpoint(config["resume_from"])
    trainer.train()


if __name__ == "__main__":
    main()
