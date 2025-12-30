import numpy as np
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d
from reflecto_exp.simulate.noise import apply_poisson_noise, get_background_noise

class XRRAugmentations:
    def __init__(
        self,
        intensity_noise_scale: float = 0.2,
        bg_range: tuple[float, float] = (-8, -5),
        res_sigma_range: tuple[float, float] = (0.0001, 0.005),
        q_shift_sigma: float = 0.002,
        delta_q: float = 0.0005,
        prob: float = 0.95,
        beam_width_range: tuple = (0.005, 0.015),
        sample_len_range: tuple = (5, 30.0),
        do_footprint: bool = True
    ):
        self.intensity_noise_scale = intensity_noise_scale
        self.bg_range = bg_range
        self.res_sigma_range = res_sigma_range
        self.q_shift_sigma = q_shift_sigma
        self.delta_q = delta_q
        self.prob = prob

        # Footprint settings
        self.beam_width_range = beam_width_range
        self.sample_len_range = sample_len_range
        self.do_footprint = do_footprint

    def _gauss_footprint(self, theta_deg: np.ndarray, beam_w: float, sample_len: float) -> np.ndarray:
        """
        GenX의 GaussIntensity 로직을 NumPy용으로 구현
        theta_deg: 입사각 (도)
        beam_w: 빔의 표준편차 (mm)
        sample_len: 시료 길이 (mm)
        """
        # GenX 로직: GaussIntensity(alpha, s1, s2, sigma_x)
        # s1 = s2 = sample_len / 2.0 (시료 중심 기준 대칭 가정)

        rad = np.pi / 180.0
        sqrt2 = np.sqrt(2.0)

        # 빔과 시료가 이루는 투영 각도 계산
        sinalpha = np.sin(theta_deg * rad)

        # 0으로 나누기 방지 (theta=0일 때)
        sinalpha = np.maximum(sinalpha, 1e-9)

        s_half = sample_len / 2.0

        # Gaussian Beam Spill-over 계산 (GenX 수식)
        # 빔이 시료 영역 [-s_half, +s_half] 안에 들어오는 비율 적분
        common = sinalpha / (sqrt2 * beam_w)

        # erf는 기함수이므로 erf(s * common) + erf(s * common) = 2 * erf(...)
        # 결과 = (2 * erf(...)) / 2.0 = erf(...)
        correction = erf(s_half * common)

        return correction

    def apply_footprint(self, q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """랜덤한 빔 폭과 시료 크기를 적용하여 저각 강도 감쇄 시뮬레이션"""
        beam_w = np.random.uniform(*self.beam_width_range)
        sample_len = np.random.uniform(*self.sample_len_range)

        # q -> theta 변환 (X-ray 파장 1.54A 기준)
        # q = 4pi/lambda * sin(theta) => theta = arcsin(q * lambda / 4pi)
        wavelength = 1.54
        sin_theta = q * wavelength / (4 * np.pi)
        # arcsin 범위 안전장치 (-1 ~ 1)
        sin_theta = np.clip(sin_theta, 0, 1)
        theta_deg = np.degrees(np.arcsin(sin_theta))

        # GenX 스타일 Gaussian Footprint 보정 계수 계산
        foocor = self._gauss_footprint(theta_deg, beam_w, sample_len)

        return R * foocor

    def __call__(self, q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 확률적으로 Augmentation 적용 여부 결정
        if np.random.rand() > self.prob:
            return q, R

        R_aug = R.copy()

        # -----------------------------------------------------------
        # [Step 1] Geometric Footprint (물리적 현상: 가장 먼저 발생)
        # -----------------------------------------------------------
        if self.do_footprint:
            R_aug = self.apply_footprint(q, R_aug)

        # -----------------------------------------------------------
        # [Step 2] Instrument Resolution (기기적 현상: 광학계 통과)
        # -----------------------------------------------------------
        # 풋프린트로 깎인 곡선이 해상도에 의해 뭉개져야 자연스러움
        res_sigma_q = np.random.uniform(*self.res_sigma_range)
        sigma_px = res_sigma_q / self.delta_q

        # Gaussian Smearing (Convolution)
        if sigma_px > 0:
            R_aug = gaussian_filter1d(R_aug, sigma=sigma_px, mode="nearest")

        # -----------------------------------------------------------
        # [Step 3] Detectors & Electronics (노이즈 및 스케일)
        # -----------------------------------------------------------

        # Global Scaling (I0 fluctuation)
        scale = np.random.uniform(1 - self.intensity_noise_scale, 1 + self.intensity_noise_scale)
        R_aug *= scale

        # Background Noise (Dark current + Scattering)
        bg_level = np.random.uniform(*self.bg_range)
        N = len(R_aug)
        bg = get_background_noise(N, bg_level, bg_level + 0.5)
        R_aug = R_aug + np.abs(bg)

        # Poisson Noise (Counting statistics - Shot noise)
        # 신호가 강한 곳은 노이즈도 큼 (Signal dependent)
        s_rand = pow(10, np.random.uniform(5, 8))
        R_aug = apply_poisson_noise(R_aug, s=s_rand)

        # -----------------------------------------------------------
        # [Step 4] Q-axis Misalignment (Calibration error)
        # -----------------------------------------------------------
        if self.q_shift_sigma > 0:
            q_shift = np.random.normal(0, self.q_shift_sigma)
            # q축이 밀리면 데이터는 반대 방향으로 이동 효과
            R_aug = np.interp(q, q + q_shift, R_aug, left=R_aug[0], right=R_aug[-1])

        return q, R_aug
