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
    print(f"Seed set to {seed}")

def ensure_data_exists(qs, config, h5_path):
    if not h5_path.exists():
        print(f"Data missing at {h5_path}. Generating...")
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        simulate.generate_1layer_data(qs, config, h5_path)

def get_dataloaders(qs, config, h5_file, stats_file):
    common_args = {
        "qs": qs, "h5_file": h5_file, "stats_file": stats_file,
        "augment": True, "min_scan_range": 0.15,
        "expand_factor": config["training"]["expand_factor"],
        "aug_prob": config["training"]["aug_prob"],
        "q_shift_sigma": config["training"]["q_shift_sigma"],
        "intensity_scale": config["training"]["intensity_scale"]
    }

    loaders = []
    for mode in ["train", "val", "test"]:
        # Only apply augment to train (handled inside Dataset logic based on mode='train')
        ds = XRR1LayerDataset(**common_args, mode=mode,
                            val_ratio=config["training"]["val_ratio"],
                            test_ratio=config["training"]["test_ratio"])
        loaders.append(DataLoader(ds, batch_size=config["training"]["batch_size"],
                        shuffle=(mode=="train"), num_workers=config["training"]["num_workers"],
                        pin_memory=torch.cuda.is_available()))
    return loaders

def main():
    print("=== EXP07: High-Res Fourier Physics Network ===")
    set_seed(42)

    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    qs_file = exp_dir / "qs.npy"

    qs = np.linspace(CONFIG["simulation"]["q_min"], CONFIG["simulation"]["q_max"],
                    CONFIG["simulation"]["q_points"])
    np.save(qs_file, qs)
    save_config(CONFIG, exp_dir / "config.json")

    ensure_data_exists(qs, CONFIG, h5_file)
    train_loader, val_loader, test_loader = get_dataloaders(qs, CONFIG, h5_file, stats_file)

    print("Initializing XRRPhysicsModel...")
    # [FIX] Fourier Config Injection
    model = XRRPhysicsModel(
        q_len=CONFIG["simulation"]["q_points"],
        input_channels=2, output_dim=3,
        n_channels=CONFIG["model"]["n_channels"],
        depth=CONFIG["model"]["depth"],
        mlp_hidden=CONFIG["model"]["mlp_hidden"],
        dropout=CONFIG["model"]["dropout"],
        use_fourier=CONFIG["model"]["use_fourier"],
        fourier_scale=CONFIG["model"]["fourier_scale"]
    )

    trainer = Trainer(model, train_loader, val_loader, exp_dir,
                    lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"],
                    patience=CONFIG["training"]["patience"])

    print("Starting Training...")
    trainer.train(CONFIG["training"]["epochs"], resume_from=exp_dir/"last.pt" if (exp_dir/"last.pt").exists() else None)

    print("Running Final Evaluation...")
    if checkpoint_file.exists():
        evaluate_pipeline(test_loader, checkpoint_file, stats_file, qs,
                        report_img_path=exp_dir/"evaluation_report.png",
                        report_csv_path=exp_dir/"evaluation_results.csv",
                        report_history_path=exp_dir/"training_history.png")

if __name__ == "__main__":
    main()
