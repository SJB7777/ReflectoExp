import torch
import torch.nn as nn


class XRR1DRegressor(nn.Module):
    """
    1D CNN으로 1-layer XRR 파라미터 회귀
    입력: (B, q_len) 반사율
    출력: (B, 3) 파라미터 (thickness, roughness, sld)
    """

    def __init__(self, q_len: int, n_channels: int = 64, depth: int = 4,
        mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()

        # 1D CNN Encoder
        layers = []
        in_ch = 1
        for i in range(depth):
            out_ch = n_channels * (2 ** min(i, 3))  # 64, 128, 256, 512
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout1d(dropout),
                nn.MaxPool1d(2)
            ])
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # MLP 회귀기
        encoder_out_dim = in_ch
        self.regressor = nn.Sequential(
            nn.Linear(encoder_out_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden // 2, 3)  # ** 3개 파라미터 직접 출력 **
        )

        self._init_weights()

    def _init_weights(self):
        """He 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, q_len) 또는 (B, 1, q_len)
        return: (B, 3)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 채널 차원 추가

        # CNN 특징 추출
        features = self.encoder(x)  # (B, ch, seq_len')
        pooled = self.global_pool(features).squeeze(-1)  # (B, ch)

        # 회귀
        return self.regressor(pooled)
