# runs/exp001_reg_tanh/main.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import RegTrainer


def main():
    # 설정
    config = {
        "exp_name": "exp001_reg_tanh",
        "h5_path": "D:/03_Resources/Data/XRR_AI/data/p300o6_raw.h5",
        "checkpoint_dir": "D:/03_Resources/Data/XRR_AI/model",
        "epochs": 3,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "val_ratio": 0.2,
        "save_every": 5,
        "model_kwargs": {
            "encoder_channels": 32,
            "encoder_depth": 4,
            "layer_hidden": 128,
            "mlp_hidden": 256,
            "dropout": 0.2,
        },
    }

    trainer = RegTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
