import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.fft
import torch.nn as nn


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


class PhysicsLayer(nn.Module):
    """
    기존 물리 모델(Fourier 기반)을 PyTorch 레이어로 감싸기
    - 입력: log10(R) 형태의 반사율
    - 출력: 두께 추정값 + 신뢰도 (0~1)
    """

    def __init__(self, q_values: torch.Tensor, n_layers: int):
        super().__init__()
        self.q_values = q_values
        self.n_layers = n_layers
        self.dq = q_values[1] - q_values[0]

    def forward(self, reflectivity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            reflectivity: (B, q_len) - 로그 반사율
        Returns:
            thickness: (B, n_layers)
            confidence: (B,) - 0~1
        """
        device = reflectivity.device
        B = reflectivity.shape[0]

        thickness = torch.zeros(B, self.n_layers, device=device)
        confidence = torch.zeros(B, device=device)

        for i in range(B):
            R = reflectivity[i].detach().cpu().numpy()
            q = self.q_values.cpu().numpy()

            thickness_est, conf = self._your_fourier_model(q, R)
            thickness[i] = torch.tensor(thickness_est, device=device)
            confidence[i] = torch.tensor(conf, device=device)

        return thickness, confidence

    # TODO: Replace the model
    def _your_fourier_model(self, q: np.ndarray, R: np.ndarray):
        """
        - 입력: q (1D), R (1D, log10 반사율)
        - 출력: thickness_est (list or ndarray, len=n_layers), confidence (float 0~1)
        """


        return thickness_est[:self.n_layers], confidence


class FusedPhysicsRegressor(nn.Module):
    def __init__(self, q_len: int, n_layers: int, q_values: torch.Tensor,
        physics_weight: float = 0.5
        ):
        super().__init__()
        self.n_layers = n_layers
        self.physics_weight = physics_weight

        # 1. 인코더 (동일)
        self.encoder = Conv1DEncoder(in_channels=1, base_channels=32, n_layers=4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 2. 개선된 물리 레이어
        self.physics_layer = PhysicsLayer(q_values, n_layers)

        # 3. Attention 기반 Fusion
        # 물리 추정치의 신뢰도에 따라 가중치 동적 조정
        self.confidence_transform = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, n_layers)  # 각 층별 신뢰도 가중치
        )

        # 4. 파라미터 예측기
        # 물리 특징 + 신경망 특징 결합
        fusion_dim = self.encoder.out_channels + n_layers * 2  # 두께 + 신뢰도
        self.param_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_layers * 3)  # 두께, 거칠기, SLD
        )

        # 5. 범위 제한 (학습 가능)
        self.thickness_range = nn.Parameter(torch.tensor([0.0, 200.0]))
        self.roughness_range = nn.Parameter(torch.tensor([0.0, 10.0]))
        self.sld_range = nn.Parameter(torch.tensor([0.0, 140.0]))

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]

        # 인코딩
        features = self.encoder(x.unsqueeze(1) if x.dim() == 2 else x)
        features = self.global_pool(features).view(batch_size, -1)

        # 물리 추정
        physics_thickness, confidence = self.physics_layer(x)

        # 신뢰도 기반 가중치 생성
        confidence_weight = self.confidence_transform(confidence.unsqueeze(1))  # (B, n_layers)
        confidence_weight = torch.sigmoid(confidence_weight) * self.physics_weight

        # Fusion 특징벡터
        # [신경망 특징, 물리 두께, 신뢰도]
        fusion_features = torch.cat([
            features,
            physics_thickness,
            confidence.unsqueeze(1).repeat(1, self.n_layers)  # 신뢰도 반복
        ], dim=1)

        # 파라미터 예측
        params = self.param_predictor(fusion_features).view(batch_size, self.n_layers, 3)
        params = torch.sigmoid(params)  # 0~1 범위

        # 역정규화
        th = params[:, :, 0] * (self.thickness_range[1] - self.thickness_range[0]) + self.thickness_range[0]
        rg = params[:, :, 1] * (self.roughness_range[1] - self.roughness_range[0]) + self.roughness_range[0]
        sld = params[:, :, 2] * (self.sld_range[1] - self.sld_range[0]) + self.sld_range[0]

        # 조건부 Fusion: 두께에만 적용
        # 신뢰도가 높을수록 물리 출력 반영
        final_th = confidence_weight * physics_thickness + (1 - confidence_weight) * th

        return {
            "physics_thickness": physics_thickness,
            "confidence": confidence,
            "nn_params": torch.stack([th, rg, sld], dim=-1),
            "final": torch.stack([final_th, rg, sld], dim=-1)
        }


def fused_loss_with_confidence(outputs, targets, physics_weight=0.5):
    """
    confidence를 가중치로 활용한 손실 함수
    """
    final_pred = outputs["final"]
    loss_final = torch.mean((final_pred - targets) ** 2)

    physics_pred = outputs["physics"]
    confidence = outputs.get("confidence", torch.ones_like(outputs["validity_mask"]))

    # confidence가 높을수록 physics loss 기여 증가
    if confidence.sum() > 0:
        loss_physics = torch.mean(
            ((physics_pred - targets[:, :, 0]) ** 2) * confidence.unsqueeze(1)
        )
    else:
        loss_physics = torch.tensor(0.0, device=targets.device)

    # 최종 손실 (confidence에 따라 physics loss 가중치 동적 조정)
    total_loss = loss_final + physics_weight * loss_physics

    return total_loss
