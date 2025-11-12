# runs/exp001_reg_tanh/model.py
import sys
from pathlib import Path

# 현재 실험 디렉토리를 최우선으로
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn


class Conv1DEncoder(nn.Module):
    """로컬 복사본 - 다른 실험과 충돌 없음"""
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


class XRRRegressor(nn.Module):
    """회귀 모델 + tanh 범위 제한 (로컬 버전)"""

    def __init__(
        self,
        q_len: int,
        n_layers: int,
        thickness_range: tuple = (0.0, 200.0),
        roughness_range: tuple = (0.0, 10.0),
        sld_range: tuple = (0.0, 140.0),
        **kwargs
    ):
        super().__init__()
        self.q_len = q_len
        self.n_layers = n_layers

        # Encoder
        self.encoder = Conv1DEncoder()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_layers * 3)
        )

        # 범위 제한 파라미터 (자동 계산된 범위 사용)
        self.register_buffer("t_min", torch.tensor(thickness_range[0]))
        self.register_buffer("t_max", torch.tensor(thickness_range[1]))
        self.register_buffer("r_min", torch.tensor(roughness_range[0]))
        self.register_buffer("r_max", torch.tensor(roughness_range[1]))
        self.register_buffer("s_min", torch.tensor(sld_range[0]))
        self.register_buffer("s_max", torch.tensor(sld_range[1]))

        # 타겟 통계 (학습 후 업데이트)
        self.register_buffer("tg_mean", torch.zeros(3))
        self.register_buffer("tg_std", torch.ones(3))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        z = self.encoder(x)
        z = self.global_pool(z).view(z.size(0), -1)

        params = self.regressor(z).view(-1, self.n_layers, 3)
        params_tanh = torch.tanh(params)

        # 범위 매핑: [-1, 1] → [min, max]
        norm = (params_tanh + 1) / 2
        th = norm[:, :, 0] * (self.t_max - self.t_min) + self.t_min
        rg = norm[:, :, 1] * (self.r_max - self.r_min) + self.r_min
        sld = norm[:, :, 2] * (self.s_max - self.s_min) + self.s_min

        return {
            "continuous": params_tanh,
            "physical": torch.stack([th, rg, sld], dim=-1)
        }

    def predict(self, x):
        return self.forward(x)["physical"]


def mse_loss(predictions, targets, weights=(1.0, 1.0, 1.0)):
    """MSE 손실 (로컬)"""
    preds = predictions["continuous"]

    loss_th = torch.mean((preds[:, :, 0] - targets[:, :, 0]) ** 2)
    loss_rg = torch.mean((preds[:, :, 1] - targets[:, :, 1]) ** 2)
    loss_sld = torch.mean((preds[:, :, 2] - targets[:, :, 2]) ** 2)

    return weights[0] * loss_th + weights[1] * loss_rg + weights[2] * loss_sld
