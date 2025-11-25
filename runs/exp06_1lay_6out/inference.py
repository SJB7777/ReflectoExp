from pathlib import Path

import torch
from config import CONFIG
from dataset import XRRPreprocessor
from xrr_model import XRR1DRegressor


class XRRInferenceEngine:
    def __init__(self, exp_dir=None):
        """
        Initializes the inference engine: loads configuration, model, and statistics.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Device: {self.device}")

        # 1. Set Path
        if exp_dir is None:
            exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
        else:
            exp_dir = Path(exp_dir)

        self.stats_file = exp_dir / "stats.pt"
        self.checkpoint_file = exp_dir / "best.pt"

        # 2. Check files
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Statistics file not found: {self.stats_file}")
        if not self.checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_file}")

        # 3. Master Grid Settings
        self.q_min = CONFIG["simulation"]["q_min"]
        self.q_max = CONFIG["simulation"]["q_max"]
        self.n_points = CONFIG["simulation"]["q_points"]

        # 4. Initialize Processor (Handles Preprocessing & Denormalization)
        self.processor = XRRPreprocessor(
            q_min=self.q_min,
            q_max=self.q_max,
            n_points=self.n_points,
            stats_file=self.stats_file,
            device=self.device
        )

        # 5. Load Model
        self._load_model()

    def _load_model(self):
        """Loads the model architecture and weights."""
        # Load checkpoint
        ckpt = torch.load(self.checkpoint_file, map_location=self.device)

        # Get model configuration (fallback to Config if not present)
        model_args = ckpt.get('config', {}).get('model_args', {
            'q_len': self.n_points,
            'input_channels': 2,  # Fixed: 2 channels (LogR, Mask)
            'n_channels': CONFIG["model"]["n_channels"],
            'depth': CONFIG["model"]["depth"],
            'mlp_hidden': CONFIG["model"]["mlp_hidden"],
        })

        # Initialize and load model
        self.model = XRR1DRegressor(**model_args).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"[Inference] Model loaded from {self.checkpoint_file}")

    def predict(self, q_raw, R_raw):
        """
        Performs inference on a single data point.

        Args:
            q_raw (np.array): Measured q values (1D).
            R_raw (np.array): Measured Reflectivity values (1D).

        Returns:
            np.array: Predicted physical parameters [thickness, roughness, sld].
        """
        # 1. Preprocess (Delegate to Processor)
        # Output Shape: (2, n_points) -> Add Batch Dimension: (1, 2, n_points)
        x = self.processor.process_input(q_raw, R_raw).unsqueeze(0).to(self.device)

        # 2. Inference
        with torch.no_grad():
            # Output: (1, 3) -> Normalized Params
            y_pred_norm = self.model(x).squeeze(0)

        # 3. Denormalize (Delegate to Processor)
        y_pred = self.processor.denormalize_params(y_pred_norm)

        return y_pred
