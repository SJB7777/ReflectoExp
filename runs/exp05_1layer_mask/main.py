from pathlib import Path

import numpy as np
import torch
from config import CONFIG
from dataset import XRR1LayerDataset
from evaluate import load_checkpoint_and_evaluate
from torch.utils.data import DataLoader
from train import Trainer
from xrr_model import XRR1DRegressor


def set_seed(seed: int = 42):
    """재현성을 위한 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def ensure_data_exists(config: dict, h5_path: Path):
    """데이터 파일이 없으면 시뮬레이션 실행"""
    if not h5_path.exists():
        print(f"데이터 파일이 없습니다: {h5_path}")
        print("시뮬레이션을 실행하여 데이터를 생성합니다...")
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        # 지연 임포트 (시뮬레이션이 필요할 때만 로드)
        import simulate
        simulate.generate_1layer_data(config, h5_path)
        print("데이터 생성 완료.")
    else:
        print(f"데이터 파일 확인됨: {h5_path}")

def get_dataloaders(config: dict, h5_file: Path, stats_file: Path):
    """Dataset 및 DataLoader 생성"""

    # 공통 인자 (Config에서 q 관련 설정 가져오기)
    dataset_kwargs = {
        "h5_file": h5_file,
        "stats_file": stats_file,
        "val_ratio": config["training"]["val_ratio"],
        "test_ratio": config["training"].get("test_ratio", 0.1),
        "q_min": config["simulation"].get("q_min", 0.0),
        "q_max": config["simulation"].get("q_max", 0.5),
        "n_points": config["simulation"].get("q_points", 200),
        "augment": True,      # 학습 시 데이터 자르기 활성화
        "aug_prob": 0.5,
        "min_scan_range": 0.15
    }

    # Dataset 인스턴스 생성
    train_set = XRR1LayerDataset(**dataset_kwargs, mode="train")
    val_set   = XRR1LayerDataset(**dataset_kwargs, mode="val")
    test_set  = XRR1LayerDataset(**dataset_kwargs, mode="test")

    # DataLoader 생성
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f"데이터셋 크기: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_loader, val_loader, test_loader

def main():
    print("=== 1-Layer XRR Regression 파이프라인 시작 ===")

    # 1. 설정 및 경로 준비
    set_seed(42)

    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    report_file = exp_dir / "error_distribution.png"

    # 2. 데이터 준비
    ensure_data_exists(CONFIG, h5_file)

    # 3. 로더 생성
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG, h5_file, stats_file)

    # 4. 모델 초기화
    print("모델 초기화 중...")
    model = XRR1DRegressor(
        q_len=CONFIG["simulation"]["q_points"],
        input_channels=2,
        n_channels=CONFIG["model"]["n_channels"],
        depth=CONFIG["model"]["depth"],
        mlp_hidden=CONFIG["model"]["mlp_hidden"],
        dropout=CONFIG["model"]["dropout"],
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {param_count:,}")

    # 5. 학습 설정 및 실행
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=exp_dir,
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )

    print("학습 시작...")
    trainer.train(CONFIG["training"]["epochs"])

    # 6. 최종 평가
    print("\n" + "="*50)
    print("최종 테스트 평가 수행")
    print("="*50)

    # 베스트 체크포인트가 없으면(학습 실패 등) 예외 처리
    if checkpoint_file.exists():
        load_checkpoint_and_evaluate(test_loader, checkpoint_file, stats_file, report_file)
    else:
        print("경고: 베스트 체크포인트 파일이 없어 평가를 건너뜁니다.")


if __name__ == "__main__":
    main()
