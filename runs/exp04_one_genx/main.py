# D:\02_Projects\Dev\X-ray_AI\Reflecto\runs\exp02_one_layer\main.py
import numpy as np
import torch
from config import CONFIG
from dataset import XRR1LayerDataset
from evaluate import load_checkpoint_and_evaluate
from torch.utils.data import DataLoader

# 재현성 보장
torch.manual_seed(42)
np.random.seed(42)
# =========================================================================

def main():
    print("1-Layer XRR Regression 파이프라인 시작")
    exp_dir = CONFIG["base_dir"] / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    report_file = exp_dir / "error_distribution.png"

    if not h5_file.exists():
        print("데이터 파일 없음, 시뮬레이션 실행...")
        h5_file.parent.mkdir(exist_ok=True)
        import simulate
        simulate.generate_1layer_data(CONFIG, h5_file)

    train_set = XRR1LayerDataset(h5_file, stats_file, mode="train", val_ratio=CONFIG["training"]["val_ratio"])
    val_set = XRR1LayerDataset(h5_file, stats_file, mode="val", val_ratio=CONFIG["training"]["val_ratio"])
    test_set = XRR1LayerDataset(h5_file, stats_file, mode="test", val_ratio=CONFIG["training"]["val_ratio"])

    train_loader = DataLoader(train_set, batch_size=CONFIG["training"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["training"]["batch_size"],
        shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=CONFIG["training"]["batch_size"],
        shuffle=False, num_workers=0)

    print(f"데이터셋: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # 모델 생성
    from model import XRR1DRegressor

    model = XRR1DRegressor(
        q_len=CONFIG["simulation"]["q_points"],
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
        save_dir=exp_dir,
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )

    # 학습 시작 (체크포인트 있으면 자동 로드)
    trainer.train(CONFIG["training"]["epochs"])

    # 최종 평가
    print("\n" + "="*50)
    print("최종 테스트 평가")
    print("="*50)

    load_checkpoint_and_evaluate(test_loader, checkpoint_file, stats_file, report_file)

if __name__ == "__main__":
    main()
