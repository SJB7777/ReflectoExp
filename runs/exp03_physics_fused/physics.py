import logging
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.signal import argrelmax
from scipy.optimize import curve_fit
from scipy import interpolate, fftpack
from scipy.signal import windows as fft_windows

logger = logging.getLogger(__name__)


def func_gauss(x, a, p, w):
    return a * np.exp(-np.log(2) * ((x - p) / (w / 2))**2)


def func_gauss3(x, a1, p1, w1, a2, p2, w2, a3, p3, w3):
    return (func_gauss(x, a1, p1, w1) +
            func_gauss(x, a2, p2, w2) +
            func_gauss(x, a3, p3, w3))


def xrr_fft(x, y, d=None, window=2, n=None, step_num: int = 1000):
    """
    FFT 변환 (XRR 분석 전용)
    """
    # 등간격 보간
    f_cubic = interpolate.interp1d(x, y, kind="cubic")
    x = np.linspace(x.min(), x.max(), step_num)
    y = f_cubic(x)

    if d is None:
        d = x[1] - x[0]

    N = len(y)
    if window == 0:
        w = np.ones(N)
    elif window == 1:
        w = fft_windows.hann(N)
    elif window == 2:
        w = fft_windows.hamming(N)
    else:
        w = fft_windows.flattop(N)

    if n is None:
        n = N

    yf = 2 / N * np.abs(fftpack.fft(w * y / np.mean(w), n=n))
    xf = fftpack.fftfreq(n, d=d)
    return xf[: n // 2], yf[: n // 2]


class PhysicsLayer(nn.Module):
    def __init__(
        self,
        q_values: torch.Tensor,
        n_layers: int = 2,
        max_total_thickness: float = 200.0,
        baseline_polyorder: int = 4,
        fft_step_num: int = 1000,
        fft_window: int = 2,
        fft_n: int = 10000,
        peak_tolerance: float = 5.0,
        min_peak_height: float = 0.0,
        min_peak_distance: float = 0.0,
        min_distance_between_peaks: float = 0.0,
    ):
        super().__init__()
        self.q_values = q_values
        self.n_layers = n_layers
        self.max_total_thickness = max_total_thickness
        self.baseline_polyorder = baseline_polyorder
        self.fft_step_num = fft_step_num
        self.fft_window = fft_window
        self.fft_n = fft_n
        self.peak_tolerance = peak_tolerance
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        self.min_distance_between_peaks = min_distance_between_peaks
        self.dq = float(q_values[1] - q_values[0])

    def forward(self, reflectivity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """✅ 반환값: (physics_output, validity_mask)"""
        B = reflectivity.shape[0]
        device = reflectivity.device
        
        q_np = self.q_values.cpu().numpy()
        R_batch_np = reflectivity.cpu().numpy()
        
        physics_output = torch.zeros(B, self.n_layers, device=device)
        validity_mask = torch.zeros(B, device=device)
        
        for i in range(B):
            fitted_params = self._process_single_with_fitting(q_np, R_batch_np[i])
            
            if fitted_params is not None:
                # ✅ 첫 피크 = 첫 번째 두께, 두 번째 피크 = 두 번째 두께
                physics_output[i, 0] = float(fitted_params[1])  # p1
                physics_output[i, 1] = float(fitted_params[4])  # p2
                validity_mask[i] = 1.0
        
        return physics_output, validity_mask

    def _process_single_with_fitting(self, q: np.ndarray, R: np.ndarray):
        """ valid_comb 로직 적용 + 피팅"""
        # 1. Baseline 제거
        R_baseline_removed = self._remove_baseline(R)
        
        # 2. FFT
        x_fft, y_fft = self._compute_fft(q, R_baseline_removed)

        # 3. 피크 검출
        upper_idx = np.searchsorted(x_fft, self.max_total_thickness)
        if upper_idx <= 10:
            return None

        peak_indices = argrelmax(y_fft[:upper_idx])[0]
        if len(peak_indices) == 0:
            return None
        print(peak_indices)
        if len(peak_indices) > 7:
            return None
        # 4. 피크 위치
        peaks = x_fft[peak_indices]
        
        # 5. ✅ valid_comb 로직으로 유효한 조합 찾기
        valid_combs = self._find_valid_combinations(peaks)
        
        # 6. 조합이 없으면 fallback: Top 3 피크
        if len(valid_combs) == 0:
            peak_heights = y_fft[peak_indices]
            top3_idx = np.argsort(-peak_heights)[:3]
            top3_peaks = x_fft[peak_indices[top3_idx]]
            
            # 중복 제거
            unique_peaks = self._select_unique_peaks(top3_peaks, y_fft, x_fft)
            if len(unique_peaks) < 3:
                return None
            valid_combs = [unique_peaks]

        # 7. 각 조합 피팅 & 최적 결과 선택
        best_params = None
        best_score = -1.0

        for comb in valid_combs:
            try:
                # 피팅 범위
                fit_min = max(0, min(comb) - 10)
                fit_max = max(comb) + 10

                fit_mask = (x_fft >= fit_min) & (x_fft <= fit_max)
                if fit_mask.sum() < 20:
                    continue

                x_fit = x_fft[fit_mask]
                y_fit = y_fft[fit_mask]

                # 초기값
                p0 = [0.1, comb[0], 5.0, 0.1, comb[1], 5.0, 0.1, comb[2], 5.0]

                # # Bounds
                # bounds_lower = [0, min(x_fit), 0, 0, min(x_fit), 0, 0, min(x_fit), 0]
                # bounds_upper = [np.inf, max(x_fit), np.inf, np.inf, max(x_fit), np.inf, np.inf, max(x_fit), np.inf]
                # bounds = (bounds_lower, bounds_upper)

                # 피팅
                popt, pcov = curve_fit(
                    func_gauss3, x_fit, y_fit, 
                    p0=p0, maxfev=2000
                )

                # 품질 점수 (covariance trace)
                score = 1.0 / (1.0 + np.trace(pcov))

                if score > best_score:
                    best_params = popt
                    best_score = score

            except Exception:
                continue
        
        return best_params

    def _find_valid_combinations(self, peaks: np.ndarray) -> list:
        """✅ 주피터 노트북 valid_comb 함수 완전 복제"""
        ln = len(peaks)
        result = []
        for i in range(ln):
            for j in range(i, ln):
                remains = set(range(ln)) - {i, j}
                for k in remains:
                    a = peaks[i]
                    b = peaks[j]
                    c = peaks[k]
                    if abs(c - a - b) < self.peak_tolerance:
                        result.append([a, b, c])
        return result

    def _select_unique_peaks(self, peaks: np.ndarray, y_fft: np.ndarray, x_fft: np.ndarray):
        """중복 제거 + 최소 간격 적용"""
        peak_heights = np.array([y_fft[np.argmin(np.abs(x_fft - p))] for p in peaks])
        sorted_idx = np.argsort(-peak_heights)
        peaks_sorted = peaks[sorted_idx]
        
        unique_peaks = []
        for p in peaks_sorted:
            if all(abs(p - up) > self.min_distance_between_peaks for up in unique_peaks):
                unique_peaks.append(p)
            if len(unique_peaks) == 3:
                break
        
        while len(unique_peaks) < 3:
            unique_peaks.append(unique_peaks[-1] + self.min_distance_between_peaks)
        
        return unique_peaks

    def _remove_baseline(self, R: np.ndarray) -> np.ndarray:
        R_log = np.log10(np.clip(R, 1e-12, None))
        window_len = len(R_log)
        R_smooth = signal.savgol_filter(R_log, window_length=window_len,
            polyorder=self.baseline_polyorder, mode='interp')
        R_baseline = 10 ** R_smooth
        R_savgol = R / (R_baseline + 1e-12)
        R_savgol = R_savgol / (R_savgol[0] + 1e-12) * R.max()
        return R_savgol

    def _compute_fft(self, q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xf, yf = xrr_fft(q, R, window=self.fft_window, n=self.fft_n, step_num=self.fft_step_num)
        x_fft = xf * 2 * np.pi
        y_fft = yf / yf[0]
        return x_fft, y_fft


if __name__ == '__main__':
    from reflecto.simulator.simulator import XRRSimulator, tth2q_wavelen

    wavelen: float = 1.54  # (nm)
    tth_min: float = 0.2   # degree
    tth_max: float = 6.0
    tth_n: int = 300
    tths: np.ndarray = np.linspace(tth_min, tth_max, tth_n)
    qs: np.ndarray = tth2q_wavelen(tths, wavelen)

    xrr_simulator = XRRSimulator(qs, 2, 1)
    thicknesses, roughnesses, slds, refl = next(xrr_simulator.make_params_refl())
    refl_tensor = torch.tensor(np.log10(refl), dtype=torch.float32).unsqueeze(0)  # (1, Q)
    qs_tensor = torch.tensor(qs, dtype=torch.float32)

    physics_layer = PhysicsLayer(qs_tensor, n_layers=2)

    # 추정 실행
    with torch.no_grad():
        est_thickness, confidence = physics_layer(refl_tensor)

    # 결과 출력
    print("\nPhysicsLayer 추정 결과:")
    print(f"추정 두께: {est_thickness[0].numpy()} Å")
    print(confidence)
    print(f"실제 두께: {thicknesses} Å")
    print(f"MAE: {np.mean(np.abs(est_thickness[0].numpy()[:len(thicknesses)] - thicknesses)):.2f} Å")
