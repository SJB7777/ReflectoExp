# runs/exp001_reg_tanh/train.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from dataset import RegressionDatasetH5
from model import XRRRegressor, mse_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class RegTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Data
        train_dataset = RegressionDatasetH5(config["h5_path"], mode="train", val_ratio=config["val_ratio"])
        val_dataset = RegressionDatasetH5(config["h5_path"], mode="val", val_ratio=config["val_ratio"])

        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

        # Model
        sample = train_dataset[0]
        q_len = sample[0].shape[0]
        n_layers = sample[1].shape[0]

        self.model = XRRRegressor(
            q_len=q_len,
            n_layers=n_layers,
            thickness_range=train_dataset.thickness_range,
            roughness_range=train_dataset.roughness_range,
            sld_range=train_dataset.sld_range,
            **config.get("model_kwargs", {}),
        ).to(self.device)

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)

        # Checkpoint directory
        self.ckpt_dir = Path(config["checkpoint_dir"]) / config["exp_name"]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for refl, targets in tqdm(self.train_loader, desc="Training", leave=False):
            refl = refl.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(refl)
            loss = mse_loss(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for refl, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                refl = refl.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(refl)
                loss = mse_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(1, self.config["epochs"] + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best.pt", epoch, val_loss)

            if epoch % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt", epoch, val_loss)

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, self.ckpt_dir / filename)
        print(f"Saved: {self.ckpt_dir / filename}")
