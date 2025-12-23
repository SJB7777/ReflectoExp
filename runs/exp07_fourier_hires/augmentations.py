import numpy as np
from scipy.ndimage import gaussian_filter1d

from reflecto_exp.simulate.noise import apply_poisson_noise, get_background_noise



class XRRAugmentations:
    def __init__(
        self,
        intensity_noise_scale: float = 0.2,
        bg_range: tuple[float, float] = (-8, -5),
        res_sigma_range: tuple[float, float] = (0.0001, 0.005),
        q_shift_sigma: float = 0.002,  # [추가] main.py와 동기화
        delta_q: float = 0.0005,
        prob: float = 0.9
    ):
        self.intensity_noise_scale = intensity_noise_scale
        self.bg_range = bg_range
        self.res_sigma_range = res_sigma_range
        self.q_shift_sigma = q_shift_sigma # [추가]
        self.delta_q = delta_q
        self.prob = prob

    def __call__(self, q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return q, R

        R_aug = R.copy()

        # 1. Resolution Smearing
        res_sigma_q = np.random.uniform(*self.res_sigma_range)
        sigma_px = res_sigma_q / self.delta_q
        R_aug = gaussian_filter1d(R_aug, sigma=sigma_px)

        # 2. Background Noise
        bg_level = np.random.uniform(*self.bg_range)
        N = len(R_aug)
        bg = get_background_noise(N, bg_level, bg_level + 0.5)
        R_aug = R_aug + np.abs(bg)

        # 3. Dynamic Poisson Noise
        s_rand = pow(10, np.random.uniform(5, 8))
        R_aug = apply_poisson_noise(R_aug, s=s_rand)

        # 4. Global Scaling
        scale = np.random.uniform(1-self.intensity_noise_scale, 1+self.intensity_noise_scale)
        R_aug *= scale

        # 5. Q-shift (Using the sigma from main.py)
        if self.q_shift_sigma > 0:
            q_shift = np.random.normal(0, self.q_shift_sigma)
            R_aug = np.interp(q, q + q_shift, R_aug, left=R_aug[0], right=R_aug[-1])

        return q, R_aug
