# main.py
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ==================== 설정 (하나의 파일에서 모두 관리) ====================
CONFIG = {
    "simulation": {
        "n_samples": 50000,  # 학습용 샘플 수
        "q_points": 200,
    },
    "model": {
        "n_channels": 64,
        "depth": 4,
        "mlp_hidden": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 128,
        "epochs": 5,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "val_ratio": 0.2,
    },
    "paths": {
        "h5_file": Path(r"D:\data\XRR_AI\one_layer") / "xrr_1layer_small.h5",
        "stats_file": Path(r"D:\data\XRR_AI\one_layer") / "stats_1layer.pt",
        "checkpoint_dir": Path(r"D:\data\XRR_AI\one_layer") / "checkpoints",
    }
}

# 재현성 보장
torch.manual_seed(42)
np.random.seed(42)
# =========================================================================

def main():
    print("🎯 1-Layer XRR Regression 파이프라인 시작")

    # 데이터 확인 및 생성
    h5_path = Path(CONFIG["paths"]["h5_file"])
    if not h5_path.exists():
        print("데이터 파일 없음, 시뮬레이션 실행...")
        import simulate
        simulate.generate_1layer_data()

    # 데이터셋 생성
    from dataset import XRR1LayerDataset

    train_set = XRR1LayerDataset(h5_path, mode="train", val_ratio=CONFIG["training"]["val_ratio"])
    val_set = XRR1LayerDataset(h5_path, mode="val", val_ratio=CONFIG["training"]["val_ratio"])
    test_set = XRR1LayerDataset(h5_path, mode="test", val_ratio=CONFIG["training"]["val_ratio"])

    train_loader = DataLoader(train_set, batch_size=CONFIG["training"]["batch_size"],
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["training"]["batch_size"],
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=CONFIG["training"]["batch_size"],
                            shuffle=False, num_workers=2)

    print(f"📊 데이터셋: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

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

    print(f"🤖 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 학습
    from train import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        checkpoint_dir=CONFIG["paths"]["checkpoint_dir"],
    )

    trainer.train(CONFIG["training"]["epochs"])

    # 최종 평가
    print("\n" + "="*50)
    print("🎯 최종 테스트 평가")
    print("="*50)

    from evaluate import load_model_and_evaluate

    checkpoint_path = Path(CONFIG["paths"]["checkpoint_dir"]) / "best.pt"
    stats_path = Path(CONFIG["paths"]["stats_file"])

    load_model_and_evaluate(checkpoint_path, h5_path, stats_path)

if __name__ == "__main__":
    main()
