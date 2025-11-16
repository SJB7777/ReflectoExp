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
        self.early_stop_patience = 15

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

    def train(self, epochs: int, resume_from: str | Path | None = None):
        """학습 시작 (이어서 학습 지원)"""
        start_epoch = 1

        # 체크포인트 로드 시도 (resume_from 우선, 없으면 best.pt 시도)
        checkpoint_path = Path(resume_from) if resume_from else self.checkpoint_dir / "best.pt"

        if checkpoint_path.exists():
            try:
                start_epoch = self.load_checkpoint(checkpoint_path) + 1
                print(f"✅ 체크포인트 로드 성공, 에포크 {start_epoch}부터 시작")
            except Exception as e:
                print(f"⚠️ 체크포인트 로드 실패: {e}")
                print("처음부터 학습을 시작합니다.")
                start_epoch = 1
        else:
            print("체크포인트 없음, 처음부터 학습 시작")

        print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")

        for epoch in range(start_epoch, epochs + 1):
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
        """체크포인트 저장 (학습 상태 포함)"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': {
                'model_class': self.model.__class__.__name__,
                'model_args': self.model.config,
            }
        }, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {self.checkpoint_dir / filename}")

    def load_checkpoint(self, filepath: str | Path) -> int:
        """체크포인트 로드 (에포크 번호 반환)"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # 모델 상태 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 옵티마이저 상태 로드 (없으면 스킵)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 스케줄러 상태 로드 (없으면 스킵)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('val_loss', float('inf'))

        loaded_epoch = checkpoint.get('epoch', 0)
        print(f"✅ 체크포인트 로드 완료: {filepath} (에포크 {loaded_epoch})")
        return loaded_epoch
