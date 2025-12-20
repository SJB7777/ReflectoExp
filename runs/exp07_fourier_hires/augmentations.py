import numpy as np
import torch

from reflecto_exp.simulate.noise import add_noise

class XRRAugmentations:
    """
    Applies physics-based data augmentation to XRR signals.
    """
    def __init__(
        self,
        intensity_noise_scale: float = 0.1,  # +/- 10% scaling
        q_shift_sigma: float = 0.002,        # Standard deviation for q-shift (1/A)
        background_level: float = 1e-7,      # Random background noise floor
        prob: float = 0.5                    # Probability of applying augmentation
    ):
        self.intensity_scale = intensity_noise_scale
        self.q_shift_sigma = q_shift_sigma
        self.background_level = background_level
        self.prob = prob

    def __call__(self, q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations.
        Args:
            q: q vector (1D numpy array)
            R: Reflectivity vector (1D numpy array)
        Returns:
            Modified (q, R)
        """
        if np.random.rand() > self.prob:
            return q, R

        # 1. Intensity Scaling (Global scaling error)
        # Simulates errors in I0 normalization
        scale_factor = np.random.uniform(1.0 - self.intensity_scale, 1.0 + self.intensity_scale)
        R_aug = R * scale_factor

        # 2. Q-Shifting (Alignment error)
        shift = np.random.normal(0, self.q_shift_sigma)
        q_shifted = q + shift
        R_aug = np.interp(q, q_shifted, R_aug, left=R_aug[0], right=R_aug[-1])

        R_aug = add_noise(R_aug)

        return q, R_aug
