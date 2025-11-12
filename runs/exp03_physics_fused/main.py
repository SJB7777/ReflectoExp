import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_fused import FusedTrainer


def main():
    config = {
        "exp_name": "exp04_physics_fused",
        "h5_path": "/mnt/data/reflecto/data/p300o6_raw.h5",
        "checkpoint_dir": "/mnt/data/reflecto/checkpoints",
        "epochs": 30,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "val_ratio": 0.2,
        "save_every": 5,
        "physics_weight": 0.5,    # 물리 모델 가중치
        "fail_safe_mode": "mask", # 실패 처리 방식: "mask", "none", "stochastic"
        "physics_dropout": 0.1,   # 확률적 드롭아웃 비율
    }

    trainer = FusedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
