from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Trainer class for XRR 1D Regressor.
    Handles training loop, validation, checkpointing, and early stopping.
    """
    def __init__(self,
        model: nn.Module, train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str | Path,
        lr: float = 1e-3, weight_decay: float = 1e-4,
        patience: int = 15
    ):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Checkpoint directory setup
        self.checkpoint_dir = Path(save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer: AdamW provides better regularization than standard Adam
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss Function: Mean Squared Error for regression
        self.criterion = nn.MSELoss()

        # Scheduler: Reduce Learning Rate when validation loss plateaus
        # Note: 'verbose' argument is deprecated in newer PyTorch versions, removed here.
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # Training State
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = patience

        print(f"🚀 Trainer initialized on {self.device}")

    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0

        # Progress bar for visual feedback
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

        for inputs, targets in pbar:
            # inputs: [Batch, 2, Length] (LogR + Mask)
            # targets: [Batch, 3] (Normalized Params)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            preds = self.model(inputs)
            loss = self.criterion(preds, targets)

            # Backward pass and Optimization
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.5f}'})

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def _validate(self) -> float:
        """Runs validation on the validation set."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(inputs)
                loss = self.criterion(preds, targets)

                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def train(self, epochs: int, resume_from: str | Path | None = None):
        """
        Main training loop.
        """
        start_epoch = 1

        # Load checkpoint if resuming
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)

        print(f"\nDataset Info: Train={len(self.train_loader.dataset)}, Val={len(self.val_loader.dataset)}")
        print("-" * 60)

        for epoch in range(start_epoch, epochs + 1):
            # 1. Training and Validation Steps
            train_loss = self._train_epoch(epoch, epochs)
            val_loss = self._validate()

            # 2. Scheduler Step
            self.scheduler.step(val_loss)

            # Log current Learning Rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # 3. Logging
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

            # 4. Save Best Model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best.pt", epoch, val_loss, is_best=True)
                print(f"  >>> New Best Model Saved! (Val Loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"  ... Patience: {self.patience_counter}/{self.early_stop_patience}")

            # 5. Early Stopping Check
            if self.patience_counter >= self.early_stop_patience:
                print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                break

            # (Optional) Save Last Model every epoch
            if epoch == epochs:
                self._save_checkpoint("last.pt", epoch, val_loss)

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float, is_best: bool = False):
        """Saves model checkpoint with metadata."""
        path = self.checkpoint_dir / filename

        # Save model configuration for easier inference later
        model_config = self.model.config if hasattr(self.model, 'config') else {}

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': {
                'model_args': model_config
            }
        }, path)

    def _load_checkpoint(self, filepath: str | Path) -> int:
        """Loads a checkpoint to resume training."""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"⚠️ Checkpoint not found: {filepath}. Starting from scratch.")
            return 1

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"✅ Checkpoint loaded: {filepath} (Resuming from Epoch {start_epoch})")
        return start_epoch
