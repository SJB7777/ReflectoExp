import numpy as np
from scipy.ndimage import gaussian_filter1d

from reflecto_exp.simulate.noise import apply_poisson_noise, get_background_noise


class XRRAugmentations:
    def __init__(
        self,
        intensity_noise_scale: float = 0.2, # I0 오차 범위 확장
        bg_range: tuple[float, float] = (-8, -5), # 바닥 노이즈 로그 범위
        res_sigma_range: tuple[float, float] = (0.0001, 0.005), # 물리적 q 단위의 Resolution sigma
        delta_q: float = 0.0005, # 포인트 간격 (pixel to q 변환용)
        prob: float = 0.9
    ):
        self.intensity_noise_scale = intensity_noise_scale
        self.bg_range = bg_range
        self.res_sigma_range = res_sigma_range
        self.delta_q = delta_q
        self.prob = prob

    def __call__(self, q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return q, R

        R_aug = R.copy()

        # 1. Random Resolution Smearing (핵심: Roughness 77A 방지)
        # 물리적 분해능 sigma_q를 픽셀 단위 sigma_px로 변환
        # sigma_px = sigma_q / delta_q
        res_sigma_q = np.random.uniform(*self.res_sigma_range)
        sigma_px = res_sigma_q / self.delta_q
        R_aug = gaussian_filter1d(R_aug, sigma=sigma_px)

        # 2. Random Background Floor 주입
        # 기기 바닥 노이즈를 먼저 깔아주어야 Poisson noise가 물리적으로 정확해짐
        bg_level = np.random.uniform(*self.bg_range)
        N = len(R_aug)
        # noise.py의 get_background_noise 활용 (로그 스케일 기반 생성 가정)
        bg = get_background_noise(N, bg_level, bg_level + 0.5)
        R_aug = R_aug + np.abs(bg)

        # 3. Dynamic Poisson Noise (Lab vs Synchrotron 모사)
        # s (광원 세기)를 고정하지 않고 10^5 ~ 10^8 사이에서 랜덤 주입
        s_rand = pow(10, np.random.uniform(5, 8))
        R_aug = apply_poisson_noise(R_aug, s=s_rand)

        # 4. Global Intensity Scaling (I0 alignment error)
        # 정규화된 데이터라도 전체적인 레벨이 미세하게 틀어지는 것을 학습
        scale = np.random.uniform(1-self.intensity_noise_scale, 1+self.intensity_noise_scale)
        R_aug *= scale

        # 5. Q-shift (Zero-point error)
        # q축 자체가 미세하게 밀리는 현상 (Alignment error) 모사
        if np.random.rand() > 0.5:
            q_shift = np.random.normal(0, 0.001)
            # q축을 실제로 밀지 않고, 고정된 q_grid 위에서 데이터만 보간으로 밀음
            R_aug = np.interp(q, q + q_shift, R_aug, left=R_aug[0], right=R_aug[-1])

        return q, R_aug
