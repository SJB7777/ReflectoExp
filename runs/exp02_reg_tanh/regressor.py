# src/reflecto/models/regressor.py
import math

import torch
import torch.nn as nn
from reflecto.models.base import Conv1DEncoder


class XRRRegressor(nn.Module):
    """
    회귀 모델: 연속값 예측 + tanh로 범위 제한
    각 파라미터 범위는 register_buffer로 모델에 포함
    """

    def __init__(
        self,
        q_len: int,
        n_layers: int,
        encoder_channels: int = 32,
        encoder_depth: int = 4,
        layer_hidden: int = 128,
        mlp_hidden: int = 256,
        dropout: float = 0.2,
        # 물리적 범위 (Simulated data 기준)
        thickness_range: tuple[float, float] = (0.0, 200.0),  # nm
        roughness_range: tuple[float, float] = (0.0, 10.0),   # Å
        sld_range: tuple[float, float] = (0.0, 140.0),      # 1e-6 Å^-2
    ):
        super().__init__()
        self.q_len = q_len
        self.n_layers = n_layers

        # Encoder (기존과 동일)
        self.encoder = Conv1DEncoder(
            in_channels=1,
            base_channels=encoder_channels,
            n_layers=encoder_depth
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        enc_out_ch = self.encoder.out_channels

        self.regressor = nn.Sequential(
            nn.Linear(enc_out_ch, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_layers * 3)  # 3=params per layer
        )

        # tanh -> 범위 매핑용 파라미터 등록
        # register_buffer: 모델 state_dict에 포함되지만 학습되지 않음
        self.register_buffer("t_min", torch.tensor(thickness_range[0]))
        self.register_buffer("t_max", torch.tensor(thickness_range[1]))
        self.register_buffer("r_min", torch.tensor(roughness_range[0]))
        self.register_buffer("r_max", torch.tensor(roughness_range[1]))
        self.register_buffer("s_min", torch.tensor(sld_range[0]))
        self.register_buffer("s_max", torch.tensor(sld_range[1]))

        # 타겟 정규화 파라미터 (학습 후 역정규화용)
        # 실제 데이터 통계로 업데이트
        self.register_buffer("tg_mean", torch.zeros(3))
        self.register_buffer("tg_std", torch.ones(3))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """학습용 타겟 정규화 파라미터 설정"""
        self.tg_mean.copy_(mean)
        self.tg_std.copy_(std)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, q_len) 또는 (B, 1, q_len)
        Returns:
            dict: {"continuous": (B, n_layers, 3), "physical": (B, n_layers, 3)}
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Encoding
        z = self.encoder(x)
        z = self.global_pool(z)
        z = z.view(z.size(0), -1)

        # Regression output (-1, 1) 범위
        params = self.regressor(z).view(-1, self.n_layers, 3)
        params_tanh = torch.tanh(params)  # [-1, 1]

        # 범위 매핑: [-1, 1] -> [0, 1] -> [min, max]
        normalized = (params_tanh + 1) / 2  # [0, 1]

        th = normalized[:, :, 0] * (self.t_max - self.t_min) + self.t_min
        rg = normalized[:, :, 1] * (self.r_max - self.r_min) + self.r_min
        sld = normalized[:, :, 2] * (self.s_max - self.s_min) + self.s_min

        return {
            "continuous": params_tanh,  # 정규화된 값 (학습용)
            "physical": torch.stack([th, rg, sld], dim=-1)  # 물리적 값 (평가용)
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """추론 시 물리적 값만 반환"""
        return self.forward(x)["physical"]


def mse_loss_with_regularization(
    predictions: dict[str, torch.Tensor],
    targets: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    range_penalty: float = 0.01  # 범위 경계 페널티
) -> torch.Tensor:
    """
    회귀 손실: MSE + 경계 페널티
    targets: (B, n_layers, 3), 이미 정규화된 값
    """
    preds = predictions["continuous"]  # tanh 출력

    # 기본 MSE
    mse_th = torch.mean((preds[:, :, 0] - targets[:, :, 0]) ** 2)
    mse_rg = torch.mean((preds[:, :, 1] - targets[:, :, 1]) ** 2)
    mse_sld = torch.mean((preds[:, :, 2] - targets[:, :, 2]) ** 2)

    total_mse = weights[0] * mse_th + weights[1] * mse_rg + weights[2] * mse_sld

    # tanh 출력이 [-1, 1] 범위를 벗어나지 않도록 페널티
    # tanh는 원래 [-1, 1]이므로 이론적으론 불필요하지만 안전장치
    range_penalty_loss = torch.mean(torch.relu(torch.abs(preds) - 0.95))

    return total_mse + range_penalty * range_penalty_loss
