from dataclasses import dataclass

import numpy as np
from refnx.reflect import SLD, ReflectModel
from refnx.reflect.structure import Stack, Structure


@dataclass
class ParamSet:
    thickness: float
    roughness: float
    sld: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.thickness, self.roughness, self.sld], dtype=np.float64)

def tth2q_wavelen[T: (float, np.ndarray)](tth: T, wavelen: float) -> T:
    """
    Convert 2θ (in degrees) and wavelength (in Å) to scattering vector q (in 1/Å).
    tth: degree
    wavelen: Å

    -> q: 1/Å
    """
    if 0 > tth > 90:
        raise ValueError(f"tth should have value under (0, 90) not {tth}")
    th_rad = np.radians(tth / 2.0)
    return (4 * np.pi / wavelen) * np.sin(th_rad)

def build_structure(params: list[ParamSet]) -> Structure:
    """refnx Structure를 연속 파라미터로부터 생성."""
    air = SLD(0.0, name="Air")
    substrate = SLD(2.0, name="Substrate")

    stack = Stack(name="MultiLayer", repeats=len(params))

    for param in params:
        mat = SLD(param.sld, name=f"SLD={param.sld}")
        stack.append(mat(param.thickness, param.roughness))

    return air(0, 0) | stack | substrate(0, 3)

def compute_reflectivity(structure: Structure, q: np.ndarray) -> np.ndarray:
    model = ReflectModel(structure)
    return model(q)

def xrr_simulate(qs, params: list[ParamSet], has_noise=False) -> np.ndarray:
    structure = build_structure(params)
    refl = compute_reflectivity(structure, qs)

    if has_noise:
        # TODO: add noise
        pass

    return refl


class ParamQuantizer:
    """
    각 파라미터마다 개별 bins를 지정하여 양자화한다.
    예: thickness_bins, roughness_bins, sld_bins
    """
    def __init__(self, thickness_bins: np.ndarray,
                 roughness_bins: np.ndarray,
                 sld_bins: np.ndarray):
        self.thickness_bins = thickness_bins
        self.roughness_bins = roughness_bins
        self.sld_bins = sld_bins

        # 라벨은 자연스럽게 구간 번호로 지정
        self.thickness_labels = np.arange(len(thickness_bins) - 1)
        self.roughness_labels = np.arange(len(roughness_bins) - 1)
        self.sld_labels = np.arange(len(sld_bins) - 1)

    @staticmethod
    def _quantize_single(value: float, bins: np.ndarray) -> int:
        idx = np.digitize(value, bins) - 1
        if idx < 0:
            idx = 0
        if idx >= len(bins) - 1:
            idx = len(bins) - 2
        return idx

    def quantize(self, param: ParamSet) -> np.ndarray:
        """
        ParamSet → np.ndarray 라벨 (예: [tl, rl, sl])
        """
        tl = self._quantize_single(param.thickness, self.thickness_bins)
        rl = self._quantize_single(param.roughness, self.roughness_bins)
        sl = self._quantize_single(param.sld,       self.sld_bins)
        return np.array([tl, rl, sl], dtype=np.int64)


class XRRPipeline:
    """
    XRR 시뮬레이션 + 양자화 + 라벨 생성 통합 파이프라인.
    기존 코드(xrr_simulate, build_structure 등)를 내부에서 호출한다.
    """

    def __init__(self, quantizer: ParamQuantizer):
        self.quantizer = quantizer

    def simulate_and_label(self, q: np.ndarray, params: list[ParamSet], has_noise=False):
        """
        params: multi-layer (list of ParamSet)
        return: dict with keys:
            - "reflectivity": np.ndarray
            - "label": np.ndarray (multi-parameter quantized label)
            - "params": list[ParamSet]
        """

        refl = xrr_simulate(q, params, has_noise=has_noise)

        # multi-layer인 경우, 전체 ParamSet을 독립적으로 quantize 후 concatenation
        all_labels = [self.quantizer.quantize(p) for p in params]
        lab = np.concatenate(all_labels, axis=0)

        return {
            "params": params,           # 원본 실수 파라미터
            "reflectivity": refl,       # XRR curve
            "label": lab,               # 양자화 라벨 (길이 = layer_count * 3)
        }
