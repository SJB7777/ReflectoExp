import json
from pathlib import Path
import gc

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.trial import TrialState

from config import CONFIG
from dataset import XRR1LayerDataset
from xrr_model import XRR1DRegressor

# ---------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------
N_TRIALS = 50            # Total number of trials
N_EPOCHS_PER_TRIAL = 15  # Epochs per trial (keep it lower than full training)
STUDY_NAME = "xrr_optimization"
STORAGE_URL = "sqlite:///xrr_tuning.db" # Saves results to a database file

def get_dataloaders(batch_size):
    """
    Creates DataLoaders with a specific batch size for the trial.
    """
    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"

    # Common args from CONFIG
    dataset_kwargs = {
        "h5_file": h5_file,
        "stats_file": stats_file,
        "val_ratio": CONFIG["training"]["val_ratio"],
        "test_ratio": CONFIG["training"]["test_ratio"],
        "q_min": CONFIG["simulation"]["q_min"],
        "q_max": CONFIG["simulation"]["q_max"],
        "n_points": CONFIG["simulation"]["q_points"],
        "augment": True,
        "aug_prob": 0.5,
        "min_scan_range": 0.15
    }

    train_set = XRR1LayerDataset(**dataset_kwargs, mode="train")
    val_set = XRR1LayerDataset(**dataset_kwargs, mode="val")

    # Important: pin_memory=False for Windows stability if needed
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

def objective(trial):
    # ---------------------------------------------------------
    # 2. Define Hyperparameter Search Space
    # ---------------------------------------------------------
    # Architecture
    n_channels = trial.suggest_categorical("n_channels", [32, 64])
    depth = trial.suggest_int("depth", 3, 5)
    mlp_hidden = trial.suggest_categorical("mlp_hidden", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Optimization
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # Keep max low to avoid OOM

    # ---------------------------------------------------------
    # 3. Setup (Model, Data, Optimizer)
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_loader, val_loader = get_dataloaders(batch_size)
    except Exception as e:
        print(f"[Error] Data loading failed: {e}")
        raise optuna.TrialPruned() from e

    model = XRR1DRegressor(
        q_len=CONFIG["simulation"]["q_points"],
        input_channels=2,
        output_dim=6,
        n_channels=n_channels,
        depth=depth,
        mlp_hidden=mlp_hidden,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")

    # ---------------------------------------------------------
    # 4. Training Loop (Simplified for Tuning)
    # ---------------------------------------------------------
    pbar = tqdm(range(N_EPOCHS_PER_TRIAL))
    for epoch in pbar:
        # --- Train ---
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                preds = model(inputs)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- Pruning (Early Stopping) ---
        # Report current result to Optuna
        trial.report(avg_val_loss, epoch)

        # Handle Pruning (if the trial is hopeless, stop it)
        if trial.should_prune():
            print(f"[Trial {trial.number}] Pruned at epoch {epoch+1} (Loss: {avg_val_loss:.4f})")
            raise optuna.TrialPruned()

    # Return the metric to minimize
    return avg_val_loss

def save_best_config(study):
    """Saves the best hyperparameters to a JSON file."""
    best_params = study.best_params
    print("\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    save_path = Path(CONFIG["base_dir"]) / CONFIG["exp_name"] / "best_hyperparameters.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"\nSaved best parameters to: {save_path}")

if __name__ == "__main__":
    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    # Define the pruner (MedianPruner is efficient for early stopping)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    # Create Study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True
    )

    print(f"Starting optimization with {N_TRIALS} trials...")

    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nTuning interrupted by user. Saving current best results...")

    # Statistics
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    if len(complete_trials) > 0:
        save_best_config(study)
    else:
        print("No trials completed successfully.")
