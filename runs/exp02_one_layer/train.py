from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
        save_dir: str | Path,
        lr: float = 1e-3, weight_decay: float = 1e-5):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=7, factor=0.5)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_dir = Path(save_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = max(10, len(self.train_loader) // 100)

        print(f"🚀 학습 시작 - Device: {self.device}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]")
        for R, params in pbar:
            R, params = R.to(self.device), params.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(R)
            loss = torch.nn.functional.mse_loss(pred, params)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return np.mean(losses)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for R, params in self.val_loader:
                R, params = R.to(self.device), params.to(self.device)
                pred = self.model(R)
                total_loss += torch.nn.functional.mse_loss(pred, params, reduction='sum').item()

        return total_loss / len(self.val_loader.dataset)

    def train(self, epochs: int):
        print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"【Epoch {epoch:02d}】 Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.scheduler.step(val_loss)

            # 체크포인트 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best.pt', epoch, val_loss)
                print("모델 저장 (best)")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.early_stop_patience}")

            if self.patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': {
                'model_class': self.model.__class__.__name__,
                'model_args': self.model.config,
            }
        }, self.checkpoint_dir / filename)
