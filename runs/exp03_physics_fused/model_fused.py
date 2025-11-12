import logging
import sys

import torch
import torch.nn as nn
from physics import PhysicsLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('physics_debug.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Conv1DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, n_layers=4, kernel_size=7):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = base_channels * (2 ** i) if i < 3 else base_channels * 8
            layers.extend([
                nn.Conv1d(ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ])
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class FusedPhysicsRegressor(nn.Module):
    def __init__(self, q_len: int, n_layers: int, q_values: torch.Tensor,
        physics_weight: float = 0.5, norm_ranges: dict | None = None):
        super().__init__()
        self.n_layers = n_layers
        self.physics_weight = physics_weight

        self.encoder = Conv1DEncoder(in_channels=1, base_channels=32, n_layers=4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.physics_layer = PhysicsLayer(q_values, n_layers, confidence_threshold=0.1)

        fusion_dim = self.encoder.out_channels + n_layers
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_layers * 3)
        )
        if norm_ranges is None:
            norm_ranges = {
                "thickness": [0.0, 200.0],
                "roughness" : [0.0, 10.0],
                "sld" : [0.0, 140.0]
            }

        self.register_buffer("th_min", torch.tensor(norm_ranges["thickness"][0]))
        self.register_buffer("th_max", torch.tensor(norm_ranges["thickness"][1]))
        self.register_buffer("r_min", torch.tensor(norm_ranges["roughness"][0]))
        self.register_buffer("r_max", torch.tensor(norm_ranges["roughness"][1]))
        self.register_buffer("s_min", torch.tensor(norm_ranges["sld"][0]))
        self.register_buffer("s_max", torch.tensor(norm_ranges["sld"][1]))

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]

        if x.dim() == 2:
            x_enc = x.unsqueeze(1)
        features = self.encoder(x_enc)
        features = self.global_pool(features).view(batch_size, -1)

        physics_output, validity_mask = self.physics_layer(x)

        combined = torch.cat([features, physics_output], dim=1)
        params = self.regressor(combined).view(batch_size, self.n_layers, 3)
        params_tanh = torch.tanh(params)
        norm = (params_tanh + 1) / 2

        th = norm[:, :, 0] * (self.th_max - self.th_min) + self.th_min
        rg = norm[:, :, 1] * (self.r_max - self.r_min) + self.r_min
        sld = norm[:, :, 2] * (self.s_max - self.s_min) + self.s_min

        mask = validity_mask.unsqueeze(1).unsqueeze(2)
        physics_weight_masked = self.physics_weight * mask

        final_th = physics_weight_masked * physics_output + (1 - physics_weight_masked) * th

        return {
            "physics": physics_output,
            "validity_mask": validity_mask,
            "nn_raw": torch.stack([th, rg, sld], dim=-1),
            "final": torch.stack([final_th, rg, sld], dim=-1),
            "confidence": validity_mask.float()
        }


def fused_loss_with_masking(outputs, targets, physics_weight=0.5):
    final_pred = outputs["final"]
    loss_final = torch.mean((final_pred - targets) ** 2)

    physics_pred = outputs["physics"]
    validity_mask = outputs["validity_mask"]

    if validity_mask.sum() > 0:
        loss_physics = torch.mean(
            ((physics_pred - targets[:, :, 0]) ** 2) *
            validity_mask.unsqueeze(1)
        )
    else:
        loss_physics = torch.tensor(0.0, device=targets.device)

    return loss_final + 0.1 * loss_physics
