from pathlib import Path

import numpy as np
import simulate
import torch
from config import CONFIG, save_config
from dataset import XRR1LayerDataset
from evaluate import evaluate_pipeline
from torch.utils.data import DataLoader
from train import Trainer
from xrr_model import XRRPhysicsModel

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    print(f"✅ Seed set to {seed}")

def ensure_data_exists(qs, config, h5_path):
    if not h5_path.exists():
        print(f"📦 Data missing at {h5_path}. Generating clean source...")
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        simulate.generate_1layer_data(qs, config, h5_path)

def get_dataloaders(qs, config, h5_file, stats_file):
    t_cfg = config["training"]

    # 우리가 수정한 Physics-Augmentation 파라미터들
    common_args = {
        "qs": qs,
        "h5_file": h5_file,
        "stats_file": stats_file,
        "augment": t_cfg.get("augment", True),
        "expand_factor": t_cfg["expand_factor"],
        "aug_prob": t_cfg["aug_prob"],
        "intensity_scale": t_cfg["intensity_scale"],
        "q_shift_sigma": t_cfg["q_shift_sigma"],
        # [핵심] 77A 방지용 Resolution Smearing 범위 (물리적 q 단위)
        "res_sigma_range": (0.0001, 0.006)
    }

    loaders = []
    for mode in ["train", "val", "test"]:
        # shuffle은 학습 데이터에만 적용
        ds = XRR1LayerDataset(**common_args, mode=mode,
                            val_ratio=t_cfg["val_ratio"],
                            test_ratio=t_cfg["test_ratio"])

        loaders.append(DataLoader(
            ds,
            batch_size=t_cfg["batch_size"],
            shuffle=(mode=="train"),
            num_workers=t_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=(mode=="train") # Batch norm 안정성
        ))
    return loaders
def main():
    print("🚀 EXP07 Launching: Physics-Informed Fourier Network")
    set_seed(42)
    # 1. 경로 및 설정 저장
    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    qs_file = exp_dir / "qs.npy"

    # 2. Grid 생성 (학습의 기준점)
    # [Tip] 재학습 시 Cubic Spline을 쓰므로 포인트를 2000개로 넉넉히 유지
    sim_cfg = CONFIG["simulation"]
    qs = np.linspace(sim_cfg["q_min"], sim_cfg["q_max"], sim_cfg["q_points"]).astype(np.float32)
    np.save(qs_file, qs)
    save_config(CONFIG, exp_dir / "config.json")
    # 3. 데이터 준비 및 로드
    ensure_data_exists(qs, CONFIG, h5_file)
    train_loader, val_loader, test_loader = get_dataloaders(qs, CONFIG, h5_file, stats_file)

    # 4. 모델 초기화
    print("🧠 Initializing Fourier Physics Network...")
    m_cfg = CONFIG["model"]
    model = XRRPhysicsModel(
        q_len=sim_cfg["q_points"],
        input_channels=2, # [LogR, Mask]
        output_dim=3,     # [Thick, Rough, SLD]
        n_channels=m_cfg["n_channels"],
        depth=m_cfg["depth"],
        mlp_hidden=m_cfg["mlp_hidden"],
        dropout=m_cfg["dropout"],
        use_fourier=m_cfg["use_fourier"],
        fourier_scale=m_cfg["fourier_scale"]
    )
    # 5. Trainer 실행
    trainer = Trainer(
        model, train_loader, val_loader, exp_dir,
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        patience=CONFIG["training"]["patience"]
    )

    print("🔥 Training Start...")
    resume_path = exp_dir / "last.pt"
    trainer.train(
        CONFIG["training"]["epochs"],
        resume_from=resume_path if resume_path.exists() else None
    )

    # 6. 최종 평가 (Physics Report 생성)
    print("\n🏁 Running Final Physics-based Evaluation...")
    if checkpoint_file.exists():
        evaluate_pipeline(
            test_loader, checkpoint_file, stats_file, qs,
            report_img_path=exp_dir / "evaluation_report.png",
            report_csv_path=exp_dir / "evaluation_results.csv",
            report_history_path=exp_dir / "training_history.png"
        )

if __name__ == "__main__":
    main()

