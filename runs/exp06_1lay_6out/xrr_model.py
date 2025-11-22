import torch
import torch.nn as nn


class XRR1DRegressor(nn.Module):
    def __init__(self, q_len: int, input_channels: int = 2, output_dim: int = 6, n_channels: int = 64, depth: int = 4,
        mlp_hidden: int = 256, dropout: float = 0.1):
        """
        XRR 1D CNN Regressor

        Args:
            q_len: Length of input q grid (n_points). Used for config logging.
            input_channels: Number of input channels (Default 2: [Reflectivity, Mask]).
            output_dim: Number of parameters to predict (Default 6: [d, sigma, sld, sio2_d, sio2_sigma, sio2_sld]).
            n_channels: Number of output channels for the first CNN layer.
            depth: Depth of the CNN encoder (number of conv blocks).
            mlp_hidden: Hidden size of the MLP regressor.
            dropout: Dropout probability.
        """
        super().__init__()
        self.config = {
            'q_len': q_len,
            'input_channels': input_channels,
            'output_dim': output_dim,
            'n_channels': n_channels,
            'depth': depth,
            'mlp_hidden': mlp_hidden,
            'dropout': dropout,
        }

        # ---------------------------------------------------------
        # 1D CNN Encoder
        # ---------------------------------------------------------
        layers = []
        in_ch = input_channels

        for i in range(depth):
            out_ch = n_channels * (2 ** min(i, 3))
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout1d(dropout),
                nn.MaxPool1d(kernel_size=2)
            ])
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # Global Average Pooling: (B, C, L) -> (B, C, 1)
        # Summarizes features across the entire q-range
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---------------------------------------------------------
        # MLP Regressor
        # ---------------------------------------------------------
        encoder_out_dim = in_ch  # The channel count of the last conv layer
        self.regressor = nn.Sequential(
            nn.Linear(encoder_out_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden // 2, output_dim)  # 출력: Thickness, Roughness, SLD
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He (Kaiming) Normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, input_channels, q_len).
                - Channel 0: Log-normalized Reflectivity
                - Channel 1: Mask (1 for valid data, 0 for padding)

        Returns:
            Predicted normalized parameters of shape (Batch, output_dim).
        """
        # 1. Extract Features
        features = self.encoder(x)  # Output: (B, Last_Ch, Reduced_Len)

        # 2. Global Pooling
        pooled = self.global_pool(features).squeeze(-1)  # Output: (B, Last_Ch)

        # 3. Regression
        return self.regressor(pooled)

