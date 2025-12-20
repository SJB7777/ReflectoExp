from pathlib import Path

import numpy as np
import simulate
import torch
from config import CONFIG, save_config
from dataset import XRR1LayerDataset
from evaluate import evaluate_pipeline
from torch.utils.data import DataLoader
from train import Trainer
from xrr_model import XRR1DRegressor

from reflecto_exp.math_utils import powerspace


def set_seed(seed: int = 42):
    """Fix random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    print(f"Seed set to {seed}")


def ensure_data_exists(qs: np.ndarray, config: dict, h5_path: Path):
    """Run simulation if data file does not exist."""
    if not h5_path.exists():
        print(f"Data file not found: {h5_path}")
        print("Running simulation to generate data...")
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        simulate.generate_1layer_data(qs, config, h5_path)
        print("Data generation complete.")

    else:
        print(f"Data file found: {h5_path}")


def get_dataloaders(qs: np.ndarray, config: dict, h5_file: Path, stats_file: Path):
    """Create Dataset and DataLoaders."""

    # Common arguments (Get q-related settings from Config)
    dataset_kwargs = {
        "qs": qs,
        "h5_file": h5_file,
        "stats_file": stats_file,
        "val_ratio": config["training"]["val_ratio"],
        "test_ratio": config["training"]["test_ratio"],
        "augment": True,
        "aug_prob": 0.5,
        "min_scan_range": 0.15
    }

    # Create Dataset instances
    train_set = XRR1LayerDataset(**dataset_kwargs, mode="train")
    val_set   = XRR1LayerDataset(**dataset_kwargs, mode="val")
    test_set  = XRR1LayerDataset(**dataset_kwargs, mode="test")

    # Create DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Dataset sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_loader, val_loader, test_loader


def main():
    print("=== 1-Layer XRR Regression Pipeline Started ===")
    if not Path(CONFIG["base_dir"]).exists():
        raise FileNotFoundError(f"Base directory {CONFIG["base_dir"]} does not exist.")
    # 1. Setup and Path Preparation
    set_seed(42)

    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    report_file_img = exp_dir / "error_distribution.png"
    report_file_csv = exp_dir / "evaluation_results.csv"
    report_history_img = exp_dir / "training_history.png"
    config_file_json = exp_dir / "config.json"
    qs: np.ndarray = powerspace(
        CONFIG["simulation"]["q_min"],
        CONFIG["simulation"]["q_max"],
        CONFIG["simulation"]["q_points"],
        CONFIG["simulation"]["power"])
    save_config(CONFIG, config_file_json)
    print(f"Config file saved at '{config_file_json}'")
    # 2. Data Preparation
    ensure_data_exists(qs, CONFIG, h5_file)

    # 3. Create Loaders
    train_loader, val_loader, test_loader = get_dataloaders(qs, CONFIG, h5_file, stats_file)

    # 4. Model Initialization
    print("Initializing model...")
    model = XRR1DRegressor(
        q_len=CONFIG["simulation"]["q_points"],
        input_channels=2,
        output_dim=6,
        n_channels=CONFIG["model"]["n_channels"],
        depth=CONFIG["model"]["depth"],
        mlp_hidden=CONFIG["model"]["mlp_hidden"],
        dropout=CONFIG["model"]["dropout"],
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # 5. Training Setup and Execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=exp_dir,
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        patience=CONFIG["training"]["patience"]
    )

    print("Starting training...")
    trainer.train(CONFIG["training"]["epochs"])

    # 6. Final Evaluation
    print("\n" + "="*50)
    print("Performing Final Test Evaluation")
    print("="*50)

    if not checkpoint_file.exists():
        print(f"No checkpoint found at {checkpoint_file}. Skipping evaluation.")
        return

    evaluate_pipeline(
        test_loader=test_loader,
        checkpoint_path=checkpoint_file,
        stats_path=stats_file,
        report_img_path=report_file_img,
        report_csv_path=report_file_csv,
        report_history_path=report_history_img
    )


if __name__ == "__main__":
    main()
