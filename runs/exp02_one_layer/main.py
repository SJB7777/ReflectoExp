from pathlib import Path

import numpy as np
import torch
from evaluate import load_checkpoint_and_evaluate
from torch.utils.data import DataLoader

# ==================== 설정 (하나의 파일에서 모두 관리) ====================
CONFIG = {
    "exp_name": "test",
    "base_dir": Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer"),
    "param_ranges": {
        "thickness": (5.0, 200.0),
        "roughness": (0.0, 10.0),
        "sld": (0.0, 140.0),
    },
    "simulation": {
        "n_samples": 30_000,
        "q_points": 200,
        "wavelength": 1.54,
        "tth_min": 1.0,
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
        "epochs": 20,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "val_ratio": 0.2,
    },

}

# 재현성 보장
torch.manual_seed(42)
np.random.seed(42)
# =========================================================================

def main():
    from reflecto.io import next_unique_file
    print("1-Layer XRR Regression 파이프라인 시작")
    exp_dir = CONFIG["base_dir"] / CONFIG["exp_name"]
    exp_dir = next_unique_file(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"

    if not h5_file.exists():
        print("데이터 파일 없음, 시뮬레이션 실행...")
        h5_file.parent.mkdir(exist_ok=True)
        import simulate
        simulate.generate_1layer_data(CONFIG, h5_file)

    # 데이터셋 생성
    from dataset import XRR1LayerDataset

    train_set = XRR1LayerDataset(h5_file, mode="train", val_ratio=CONFIG["training"]["val_ratio"])
    val_set = XRR1LayerDataset(h5_file, mode="val", val_ratio=CONFIG["training"]["val_ratio"])
    test_set = XRR1LayerDataset(h5_file, mode="test", val_ratio=CONFIG["training"]["val_ratio"])

    train_loader = DataLoader(train_set, batch_size=CONFIG["training"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["training"]["batch_size"],
        shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=CONFIG["training"]["batch_size"],
        shuffle=False, num_workers=0)

    print(f"데이터셋: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # 모델 생성
    from model import XRR1DRegressor

    q_len = train_set.q_values.shape[0]
    model = XRR1DRegressor(
        q_len=q_len,
        n_channels=CONFIG["model"]["n_channels"],
        depth=CONFIG["model"]["depth"],
        mlp_hidden=CONFIG["model"]["mlp_hidden"],
        dropout=CONFIG["model"]["dropout"],
    )

    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 학습
    from train import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        checkpoint_dir=exp_dir,
    )

    trainer.train(CONFIG["training"]["epochs"])

    # 최종 평가
    print("\n" + "="*50)
    print("최종 테스트 평가")
    print("="*50)

    load_checkpoint_and_evaluate(checkpoint_file, test_loader, stats_file)

if __name__ == "__main__":
    main()
