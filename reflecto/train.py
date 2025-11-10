import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from reflecto.dataset import DatasetH5, ParamQuantizer
from reflecto.model import XRRClassifier, validate_epoch


class XRRTrainer:
    def __init__(
        self,
        h5_path: str,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | torch.device = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "./checkpoints",
        scheduler_step: int = 5,
        scheduler_gamma: float = 0.5,
    ):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Quantizer
        self.quantizer = ParamQuantizer(thickness_bins=None, roughness_bins=None, sld_bins=None)
        self.n_th = len(self.quantizer.thickness_bins) - 1
        self.n_rg = len(self.quantizer.roughness_bins) - 1
        self.n_sld = len(self.quantizer.sld_bins) - 1

        # Dataset
        self.train_dataset = DatasetH5(self.h5_path, self.quantizer, device="cpu")
        self.val_dataset = DatasetH5(self.h5_path, self.quantizer, device="cpu")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)

        # Model
        sample_refl, sample_label = self.train_dataset[0]
        self.q_len = sample_refl.shape[0]
        self.n_layers = sample_label.shape[0]
        print(f"Detected q_len: {self.q_len}, n_layers: {self.n_layers}")

        self.model = XRRClassifier(
            q_len=self.q_len,
            n_layers=self.n_layers,
            n_th_bins=self.n_th,
            n_rg_bins=self.n_rg,
            n_sld_bins=self.n_sld
        ).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        self.ce = nn.CrossEntropyLoss()

    def train(self, save_every: int = 5):
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            last_parts = {"th": 0.0, "rg": 0.0, "sld": 0.0}

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
            for refl, labels in pbar:
                refl = refl.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                th_logits, rg_logits, sld_logits = self.model(refl)

                tgt_th = labels[:, :, 0]
                tgt_rg = labels[:, :, 1]
                tgt_sld = labels[:, :, 2]

                loss_th = self.ce(th_logits.view(-1, th_logits.shape[-1]), tgt_th.view(-1))
                loss_rg = self.ce(rg_logits.view(-1, rg_logits.shape[-1]), tgt_rg.view(-1))
                loss_sld = self.ce(sld_logits.view(-1, sld_logits.shape[-1]), tgt_sld.view(-1))

                loss = loss_th + loss_rg + loss_sld
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                last_parts = {"th": loss_th.item(), "rg": loss_rg.item(), "sld": loss_sld.item()}

                # tqdm에 실시간 표시
                pbar.set_postfix({**last_parts, "loss": f"{loss.item():.4f}"})

            avg_train_loss = total_loss / len(self.train_loader)

            # Validation
            val_loss, val_accs = validate_epoch(self.model, self.val_loader, self.device)

            print(
                f"[Epoch {epoch:02d}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc(th/rg/sld): "
                f"{val_accs['th']:.3f} / {val_accs['rg']:.3f} / {val_accs['sld']:.3f}"
            )

            # Scheduler step
            self.scheduler.step()

            # Checkpoints
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(self.checkpoint_dir, "best_model.pt"))
            if epoch % save_every == 0:
                self.save_model(os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt"))

    def save_model(self, path="xrr_model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    h5_path = r"D:\03_Resources\Data\XRR_AI\data\p300o6_raw.h5"
    trainer = XRRTrainer(h5_path, batch_size=32, epochs=20)
    trainer.train(save_every=5)
