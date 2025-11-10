"""
Improved model utilities:
- Conv1DEncoder and XRRClassifier unchanged in structure but cleaned.
- train_epoch supports AMP via GradScaler, tqdm, gradient clip, and returns parts.
- validate_epoch uses inference_mode and checks for NaN/Inf and clamps logits.
"""

import math

import torch
import torch.nn as nn
from tqdm import tqdm


class Conv1DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, n_layers: int = 4, kernel_size: int = 7):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = base_channels * (2 ** i) if i < 3 else base_channels * (2 ** 3)
            layers.append(nn.Conv1d(ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        # input shape: (B, Q) or (B, 1, Q)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class XRRClassifier(nn.Module):
    def __init__(
        self,
        q_len: int,
        n_layers: int,
        n_th_bins: int,
        n_rg_bins: int,
        n_sld_bins: int,
        encoder_channels: int = 32,
        encoder_depth: int = 4,
        layer_hidden: int = 128,
        mlp_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.q_len = q_len
        self.n_layers = n_layers

        self.encoder = Conv1DEncoder(in_channels=1, base_channels=encoder_channels, n_layers=encoder_depth)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        enc_out_ch = self.encoder.out_channels

        self.bottleneck = nn.Sequential(
            nn.Linear(enc_out_ch, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_layers * layer_hidden),
        )
        self.layer_hidden = layer_hidden

        # heads
        self.th_head = nn.Sequential(
            nn.Linear(layer_hidden, layer_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(layer_hidden // 2, n_th_bins),
        )
        self.rg_head = nn.Sequential(
            nn.Linear(layer_hidden, layer_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(layer_hidden // 2, n_rg_bins),
        )
        self.sld_head = nn.Sequential(
            nn.Linear(layer_hidden, layer_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(layer_hidden // 2, n_sld_bins),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        z = self.global_pool(z)
        z = z.view(z.size(0), -1)

        per_layer = self.bottleneck(z)
        per_layer = per_layer.view(-1, self.n_layers, self.layer_hidden)

        B, L, H = per_layer.shape
        flat = per_layer.view(B * L, H)

        logits_th = self.th_head(flat).view(B, L, -1)
        logits_rg = self.rg_head(flat).view(B, L, -1)
        logits_sld = self.sld_head(flat).view(B, L, -1)

        return logits_th, logits_rg, logits_sld


# -------------------------
# Loss helper
# -------------------------
def multitask_loss(
    logits_th, logits_rg, logits_sld,
    targets,  # (B, L, 3)
    weights=(1.0, 1.0, 1.0),
    ignore_index: int = -100
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    B, L, _ = targets.shape
    device = logits_th.device

    logits_th_f = logits_th.view(B * L, -1)
    logits_rg_f = logits_rg.view(B * L, -1)
    logits_sld_f = logits_sld.view(B * L, -1)

    tgt_th = targets[:, :, 0].reshape(B * L).to(device)
    tgt_rg = targets[:, :, 1].reshape(B * L).to(device)
    tgt_sld = targets[:, :, 2].reshape(B * L).to(device)

    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss_th = ce(logits_th_f, tgt_th)
    loss_rg = ce(logits_rg_f, tgt_rg)
    loss_sld = ce(logits_sld_f, tgt_sld)

    total = weights[0] * loss_th + weights[1] * loss_rg + weights[2] * loss_sld
    return total, (loss_th.item(), loss_rg.item(), loss_sld.item())


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()


# -------------------------
# Training / Validation
# -------------------------
def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, scaler: torch.cuda.amp.GradScaler = None,
                clip_grad: float = 1.0) -> float:
    """
    Single epoch training with optional AMP scaler.
    Returns: (avg_loss, {"th":.., "rg":.., "sld":..})
    """
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    last_parts = {"th": 0.0, "rg": 0.0, "sld": 0.0}

    pbar = tqdm(loader, desc="Training", leave=False)
    for refl, labels in pbar:
        refl = refl.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits_th, logits_rg, logits_sld = model(refl)
                loss_th = ce(logits_th.view(-1, logits_th.shape[-1]), labels[:, :, 0].view(-1))
                loss_rg = ce(logits_rg.view(-1, logits_rg.shape[-1]), labels[:, :, 1].view(-1))
                loss_sld = ce(logits_sld.view(-1, logits_sld.shape[-1]), labels[:, :, 2].view(-1))
                loss = loss_th + loss_rg + loss_sld
            scaler.scale(loss).backward()
            # optional grad clip
            if clip_grad is not None and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_th, logits_rg, logits_sld = model(refl)
            loss_th = ce(logits_th.view(-1, logits_th.shape[-1]), labels[:, :, 0].view(-1))
            loss_rg = ce(logits_rg.view(-1, logits_rg.shape[-1]), labels[:, :, 1].view(-1))
            loss_sld = ce(logits_sld.view(-1, logits_sld.shape[-1]), labels[:, :, 2].view(-1))
            loss = loss_th + loss_rg + loss_sld
            loss.backward()
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()
        last_parts = {"th": loss_th.item(), "rg": loss_rg.item(), "sld": loss_sld.item()}

        pbar.set_postfix({**last_parts, "loss": f"{loss.item():.4f}"})

    avg = total_loss / len(loader) if len(loader) > 0 else float("nan")
    return avg


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    logits_clip: float = 50.0
) -> tuple[float, dict[str, float]]:

    model.eval()
    total_loss = 0.0
    total_items = 0
    accs = {"th": 0.0, "rg": 0.0, "sld": 0.0}

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Validation", leave=False)

        for refl, labels in pbar:
            refl = refl.to(device)
            labels = labels.to(device)

            logits_th, logits_rg, logits_sld = model(refl)

            # --- 안정성 확보: logits clipping ---
            if logits_clip is not None:
                logits_th = logits_th.clamp(-logits_clip, logits_clip)
                logits_rg = logits_rg.clamp(-logits_clip, logits_clip)
                logits_sld = logits_sld.clamp(-logits_clip, logits_clip)

            if not (torch.isfinite(logits_th).all() and 
                    torch.isfinite(logits_rg).all() and 
                    torch.isfinite(logits_sld).all()):
                raise RuntimeError("Non-finite logits encountered in validation.")

            loss, _ = multitask_loss(logits_th, logits_rg, logits_sld, labels)

            bs = refl.size(0)
            total_loss += float(loss.item()) * bs
            total_items += bs

            accs["th"] += accuracy_from_logits(logits_th, labels[..., 0]) * bs
            accs["rg"] += accuracy_from_logits(logits_rg, labels[..., 1]) * bs
            accs["sld"] += accuracy_from_logits(logits_sld, labels[..., 2]) * bs

            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_items
    for k in accs:
        accs[k] /= total_items

    return avg_loss, accs
