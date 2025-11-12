import logging
from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn
from scipy import fftpack, signal
from scipy.optimize import curve_fit
from scipy.signal import argrelmax, savgol_filter

logger = logging.getLogger(__name__)

def func_gauss(x, a, p, w):
    return a * np.exp(-np.log(2) * ((x - p) / (w / 2))**2)

def func_gauss3(x, a1, p1, w1, a2, p2, w2, a3, p3, w3):
    return func_gauss(x, a1, p1, w1) + func_gauss(x, a2, p2, w2) + func_gauss(x, a3, p3, w3)


class PhysicsLayer(nn.Module):
    def __init__(self, q_values: torch.Tensor, n_layers: int = 2,
                 confidence_threshold: float = 0.1, x_upper_bound: float = 200.0,
                 min_peak_height: float = 0.005, min_peak_distance: float = 1.0,
                 combination_tolerance: float = 10.0):
        super().__init__()
        self.q_values = q_values
        self.n_layers = n_layers
        self.confidence_threshold = confidence_threshold
        self.x_upper_bound = x_upper_bound
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        self.combination_tolerance = combination_tolerance
        self.dq = q_values[1] - q_values[0]
        self.snr_threshold = 5.0

    def forward(self, reflectivity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            B = reflectivity.shape[0]
            device = reflectivity.device
            # 배치 전체를 한 번만 CPU로 이동
            q_np = self.q_values.cpu().numpy()
            R_batch_np = reflectivity.cpu().numpy()  # (B, Q)

            # 결과 버퍼
            thickness = torch.zeros(B, self.n_layers, device=device)
            validity_mask = torch.zeros(B, device=device)

            # 로깅 최소화: 배치가 작을 때만 디버그
            debug_mode = B <= 8

            for i in range(B):
                # 샘플 인덱싱만 수행
                thick_est, conf = self._estimate_thickness_combination(
                    q_np, R_batch_np[i],
                    sample_idx=i if debug_mode else None  # 디버그 모드에서만 로깅
                )

                if len(thick_est) > 0:
                    n_valid = min(len(thick_est), self.n_layers)
                    thickness[i, :n_valid] = torch.tensor(
                        thick_est[:n_valid],
                        device=device,
                        dtype=torch.float32
                    )

                if conf > self.confidence_threshold:
                    validity_mask[i] = 1.0

            if debug_mode:
                success_rate = validity_mask.mean().item()
                logger.info(f"\n{'='*60}")
                logger.info(f"Batch Summary: Success rate = {success_rate:.2%}")
                logger.info(f"{'='*60}\n")

            return thickness, validity_mask

    def _estimate_thickness_combination(self, q: np.ndarray, R_log: np.ndarray, sample_idx=None):
        """내부 로직은 동일 (성능만 최적화)"""
        try:
            x_fft, y_fft_norm, detected_peaks = self._fft_and_detect_peaks(q, R_log, sample_idx)

            if len(detected_peaks) == 0:
                return np.array([]), 0.0

            thickness_candidates, peaks_for_fitting = self._analyze_combinations(detected_peaks, sample_idx)

            if len(thickness_candidates) == 0:
                return np.array([]), 0.0

            fit_result = self._fit_gaussians_dynamic(x_fft, y_fft_norm, peaks_for_fitting, sample_idx)
            confidence = self._calculate_confidence(fit_result, detected_peaks, thickness_candidates, sample_idx)
            final_thickness = self._select_best_thickness(thickness_candidates, confidence, sample_idx)

            return final_thickness, confidence

        except Exception as e:
            if sample_idx is not None:
                logger.error(f"[Sample {sample_idx}] Exception: {str(e)}", exc_info=True)
            return np.array([]), 0.0

    def _fft_and_detect_peaks(self, q: np.ndarray, R_log: np.ndarray, sample_idx=None):
        """FFT 및 피크 검출 (개별 샘플)"""
        R_original = 10 ** R_log
        window_len = min(len(R_log), 51)
        R_smooth = savgol_filter(R_log, window_length=window_len, polyorder=4, mode='interp')
        R_savgol = R_original / (10 ** R_smooth)
        R_savgol = self._normalize_by_R(R_savgol, R_original)

        y = R_savgol
        N = len(y)
        dq = q[1] - q[0]
        w = signal.windows.hamming(N)
        yf = 2 / N * np.abs(fftpack.fft(w * y / np.mean(w), n=10000))
        xf = fftpack.fftfreq(10000, d=dq)
        xf, yf = xf[:5000], yf[:5000]

        x_fft = xf * 2 * np.pi
        y_fft_norm = yf / (yf[0] + 1e-12)

        upper_idx = np.searchsorted(x_fft, self.x_upper_bound)
        if upper_idx <= 10:
            return x_fft, y_fft_norm, np.array([])

        detected_peaks = np.array([])
        for order in [50, 30, 20, 10]:
            idx_max = argrelmax(y_fft_norm[:upper_idx], order=order)[0]
            if len(idx_max) > 0:
                heights = y_fft_norm[idx_max]
                idx_strong = idx_max[heights > self.min_peak_height]

                final_peaks = []
                for idx in idx_strong:
                    x_pos = x_fft[idx]
                    if all(abs(x_pos - fp) > self.min_peak_distance for fp in final_peaks):
                        final_peaks.append(x_pos)

                if len(final_peaks) > 0:
                    detected_peaks = np.array(final_peaks)
                    break

        return x_fft, y_fft_norm, detected_peaks

    def _analyze_combinations(self, peaks: np.ndarray, sample_idx=None):
        if len(peaks) < 2:
            return np.array([]), np.array([])

        peaks = np.sort(peaks)
        thickness_candidates = []
        confidence_scores = []

        peaks_for_fitting = peaks[-3:] if len(peaks) >= 3 else peaks

        if sample_idx is not None:
            logger.debug(f"[Sample {sample_idx}]   Detected peaks: {peaks}")
            logger.debug(f"[Sample {sample_idx}]   Top3 for fitting: {peaks_for_fitting}")

        if len(peaks) >= 3:
            top3 = peaks[-3:]
            small_peaks = peaks[:-3]

            if len(small_peaks) >= 2:
                for combo in combinations_with_replacement(small_peaks, 2):
                    combo_sum = sum(combo)
                    for large_peak in top3:
                        error = abs(combo_sum - large_peak) / large_peak * 100
                        if error < self.combination_tolerance:
                            avg_thickness = (combo[0] + combo[1]) / 2
                            thickness_candidates.append(avg_thickness)
                            confidence_scores.append(1.0 - error / 100)

        unique_candidates = []
        unique_scores = []
        for tc, score in zip(thickness_candidates, confidence_scores, strict=True):
            if not any(abs(tc - ut) < 1.0 for ut in unique_candidates):
                unique_candidates.append(tc)
                unique_scores.append(score)

        if len(unique_candidates) > 2:
            sorted_idx = np.argsort(-np.array(unique_scores))
            result = np.array(unique_candidates)[sorted_idx[:2]]
        elif len(unique_candidates) == 1:
            result = np.array([unique_candidates[0], 0.0])
        else:
            result = np.array(unique_candidates)

        return result, peaks_for_fitting

    def _fit_gaussians_dynamic(self, x, y, peaks, sample_idx=None):
        n_peaks = min(len(peaks), 3)

        if sample_idx is not None:
            logger.debug(f"[Sample {sample_idx}]   Starting Gaussian fitting for {n_peaks} peaks...")

        if n_peaks >= 3:
            result = self._fit_3gaussians(x, y, peaks[:3], sample_idx)
            if not result['success']:
                logger.warning(f"[Sample {sample_idx}]   3-Gaussian failed → Trying 2-Gaussian")
                result = self._fit_2gaussians(x, y, peaks[:2], sample_idx)
        elif n_peaks == 2:
            result = self._fit_2gaussians(x, y, peaks[:2], sample_idx)
        elif n_peaks == 1:
            result = self._fit_1gaussian(x, y, peaks[0], sample_idx)
        else:
            return {'success': False}

        return result

    def _fit_3gaussians(self, x, y, peaks, sample_idx=None):
        try:
            p0 = [0.1, peaks[0], 5, 0.1, peaks[1], 5, 0.1, peaks[2], 5]
            bounds = ([0, min(x), 0]*3, [np.inf, max(x), np.inf]*3)
            popt, pcov = curve_fit(func_gauss3, x, y, p0=p0, bounds=bounds, maxfev=5000)

            if sample_idx is not None:
                logger.debug(f"[Sample {sample_idx}]   3-Gaussian fit: SUCCESS")

            return {'success': True, 'params': popt, 'n_peaks': 3, 'pcov': pcov}
        except Exception as e:
            if sample_idx is not None:
                logger.error(f"[Sample {sample_idx}]   3-Gaussian fit: FAILED - {str(e)}")
            return {'success': False}

    def _fit_2gaussians(self, x, y, peaks, sample_idx=None):
        def func_gauss2(x, a1, p1, w1, a2, p2, w2):
            return func_gauss(x, a1, p1, w1) + func_gauss(x, a2, p2, w2)

        try:
            p0 = [0.1, peaks[0], 5, 0.1, peaks[1], 5]
            bounds = ([0, min(x), 0]*2, [np.inf, max(x), np.inf]*2)
            popt, pcov = curve_fit(func_gauss2, x, y, p0=p0, bounds=bounds, maxfev=5000)

            if sample_idx is not None:
                logger.debug(f"[Sample {sample_idx}]   2-Gaussian fit: SUCCESS")

            params_3 = np.array([popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], 0, 0, 0])
            return {'success': True, 'params': params_3, 'n_peaks': 2, 'pcov': pcov}
        except Exception as e:
            if sample_idx is not None:
                logger.error(f"[Sample {sample_idx}]   2-Gaussian fit: FAILED - {str(e)}")
            return {'success': False}

    def _fit_1gaussian(self, x, y, peak, sample_idx=None):
        def func_gauss1(x, a1, p1, w1):
            return func_gauss(x, a1, p1, w1)

        try:
            p0 = [0.1, peak, 5]
            bounds = ([0, min(x), 0], [np.inf, max(x), np.inf])
            popt, pcov = curve_fit(func_gauss1, x, y, p0=p0, bounds=bounds, maxfev=5000)

            if sample_idx is not None:
                logger.debug(f"[Sample {sample_idx}]   1-Gaussian fit: SUCCESS")

            params_3 = np.array([popt[0], popt[1], popt[2], 0, 0, 0, 0, 0, 0])
            return {'success': True, 'params': params_3, 'n_peaks': 1, 'pcov': pcov}
        except Exception as e:
            if sample_idx is not None:
                logger.error(f"[Sample {sample_idx}]   1-Gaussian fit: FAILED - {str(e)}")
            return {'success': False}

    def _calculate_confidence(self, fit_result, detected_peaks, thickness_candidates, sample_idx=None):
        if not fit_result['success']:
            if sample_idx is not None:
                logger.warning(f"[Sample {sample_idx}]   Confidence: FIT FAILED → 0.0")
            return 0.0

        snr_conf = 0.8
        combo_conf = self._evaluate_combination_match(detected_peaks, thickness_candidates, sample_idx)
        fit_conf = self._evaluate_fit_quality(fit_result, sample_idx)
        count_weight = {3: 1.0, 2: 0.7, 1: 0.4}.get(len(detected_peaks), 0.0)

        confidence = snr_conf * combo_conf * fit_conf * count_weight

        return float(np.clip(confidence, 0, 1))

    def _evaluate_combination_match(self, peaks, thickness_candidates, sample_idx=None):
        if len(peaks) < 2 or len(thickness_candidates) == 0:
            return 0.5
        match_scores = []
        for tc in thickness_candidates:
            closest_peak = peaks[np.argmin(np.abs(peaks - tc))]
            match_score = 1 - abs(tc - closest_peak) / (closest_peak + 1e-6)
            match_scores.append(max(0, match_score))
        return np.mean(match_scores) if match_scores else 0.5

    def _evaluate_fit_quality(self, fit_result, sample_idx=None):
        try:
            pcov = fit_result.get('pcov', None)
            if pcov is None:
                return 0.7
            param_std = np.sqrt(np.diag(pcov)).mean()
            quality = np.exp(-param_std / 20)
            return float(np.clip(quality, 0.5, 1.0))
        except:
            return 0.7

    def _select_best_thickness(self, thickness_candidates, confidence, sample_idx=None):
        if len(thickness_candidates) == 0:
            return np.array([])
        if confidence > self.confidence_threshold:
            if len(thickness_candidates) >= 2:
                return np.array([thickness_candidates[0], thickness_candidates[1]])
            else:
                return np.array([thickness_candidates[0], 0.0])
        return np.array([])

    @staticmethod
    def _normalize_by_R(arr, R):
        if R.max() == 0:
            return arr
        return arr / (arr[0] + 1e-12) * R.max()
