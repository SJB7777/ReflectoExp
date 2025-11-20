import torch
import torch.nn as nn


class XRR1DRegressor(nn.Module):
    def __init__(self, q_len: int, input_channels: int = 2, n_channels: int = 64, depth: int = 4,
                 mlp_hidden: int = 256, dropout: float = 0.1):
        """
        XRR 1D CNN Regressor

        Args:
            q_len: 입력 데이터 길이 (n_points)
            input_channels: 입력 채널 수 (기본 2: [Reflectivity, Mask])
            n_channels: 첫 번째 CNN 레이어의 출력 채널 수
            depth: CNN 깊이
            mlp_hidden: MLP 은닉층 크기
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.config = {
            'q_len': q_len,
            'input_channels': input_channels,
            'n_channels': n_channels,
            'depth': depth,
            'mlp_hidden': mlp_hidden,
            'dropout': dropout,
        }

        # ---------------------------------------------------------
        # 1D CNN Encoder
        # ---------------------------------------------------------
        layers = []

        # [핵심 수정] 초기 입력 채널을 데이터셋에 맞춰 설정 (기본 2)
        in_ch = input_channels

        for i in range(depth):
            out_ch = n_channels * (2 ** min(i, 3))  # 64, 128, 256, 512...
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout1d(dropout),
                nn.MaxPool1d(2)
            ])
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # Global pooling (Feature Map 전체 평균 -> 위치 정보 소거 및 요약)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---------------------------------------------------------
        # MLP Regressor
        # ---------------------------------------------------------
        encoder_out_dim = in_ch
        self.regressor = nn.Sequential(
            nn.Linear(encoder_out_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden // 2, 3)  # 출력: Thickness, Roughness, SLD
        )

        self._init_weights()

    def _init_weights(self):
        """He 초기화 (Kaiming Normal)"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 2, q_len) - [LogR, Mask]
        Returns:
            (Batch, 3) - [Thickness, Roughness, SLD] (Normalized)
        """
        # [수정] Dataset이 이미 (Batch, 2, Length)를 보장하므로 unsqueeze 로직 제거
        # 만약 2D (Batch, Length)가 들어오면 에러가 나야 정상 (Mask 채널 누락)

        features = self.encoder(x)               # (B, Last_Ch, Reduced_Len)
        pooled = self.global_pool(features).squeeze(-1)  # (B, Last_Ch)

        return self.regressor(pooled)
