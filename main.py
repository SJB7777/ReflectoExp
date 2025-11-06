from dataclasses import dataclass

import numpy as np
from refnx.reflect import SLD, ReflectModel
from refnx.reflect.structure import Stack, Structure


# ---------------------------------------------------
# 1. 데이터 클래스: 연속 파라미터 저장용
# ---------------------------------------------------
@dataclass
class LayerParams:
    thickness: float
    roughness: float
    sld: float


@dataclass
class XRRParams:
    layers: list[LayerParams]


# ---------------------------------------------------
# 2. 유틸 함수들 (함수형 스타일)
# ---------------------------------------------------
def build_structure(layer_params: XRRParams) -> Structure:
    """refnx Structure를 연속 파라미터로부터 생성."""
    air = SLD(0.0, name="Air")
    substrate = SLD(2.0, name="Substrate")

    stack = Stack(name="MultiLayer", repeats=len(layer_params.layers))

    for lp in layer_params.layers:
        mat = SLD(lp.sld, name=f"SLD={lp.sld}")
        stack.append(mat(lp.thickness, lp.roughness))

    return air(0, 0) | stack | substrate(0, 3)


def compute_reflectivity(structure: Structure, q: np.ndarray) -> np.ndarray:
    model = ReflectModel(structure, bkg=0.0, dq=5.0)
    return model(q)


def quantize(value: float, bins: np.ndarray) -> int:
    """연속값 → 양자화 인덱스"""
    return np.digitize(value, bins) - 1


# ---------------------------------------------------
# 3. 전체 파이프라인을 묶는 클래스
# ---------------------------------------------------
class XRRDatasetGenerator:
    def __init__(
        self,
        q_range: tuple[float, float] = (0.005, 0.25),
        q_points: int = 1000,
        thickness_bins: np.ndarray = None,
        sld_bins: np.ndarray = None,
        roughness_bins: np.ndarray = None,
    ):
        self.q = np.linspace(*q_range, q_points)

        self.thickness_bins = thickness_bins
        self.roughness_bins = roughness_bins
        self.sld_bins = sld_bins

        if self.thickness_bins is None:
            self.thickness_bins = np.linspace(0, 2000, 51)
        if self.roughness_bins is None:
            self.roughness_bins = np.linspace(0, 10, 21)
        if self.sld_bins is None:
            self.sld_bins = np.linspace(1.0, 6.0, 26)


    # ------------------------------------------------
    # 3-1. 레이어 생성 (연속값)
    # ------------------------------------------------
    def sample_layer_parameters(
        self,
        thickness_range=(5, 1500),      # nm, 최대 1 µm 이상도 지원
        roughness_range=(0.1, 5),
        sld_range=(1.0, 6.0),
        n_layers: int = 1,
    ) -> XRRParams:

        layers = []
        for _ in range(n_layers):
            t = np.random.uniform(*thickness_range)
            r = np.random.uniform(*roughness_range)
            s = np.random.uniform(*sld_range)
            layers.append(LayerParams(t, r, s))

        return XRRParams(layers)

    # ------------------------------------------------
    # 3-2. 라벨 양자화 수행
    # ------------------------------------------------
    def quantize_labels(self, params: XRRParams) -> dict[str, list[float]]:
        """층마다 독립적으로 양자화."""
        quantized = {
            "thickness": [],
            "roughness": [],
            "sld": []
        }

        for lp in params.layers:
            quantized["thickness"].append(quantize(lp.thickness, self.thickness_bins))
            quantized["roughness"].append(quantize(lp.roughness, self.roughness_bins))
            quantized["sld"].append(quantize(lp.sld, self.sld_bins))

        return quantized

    # ------------------------------------------------
    # 3-3. (중심) 데이터 1개의 XRR 시뮬레이션 수행
    # ------------------------------------------------
    def generate_sample(self, params: XRRParams) -> tuple[np.ndarray, dict]:
        structure = build_structure(params)
        refl = compute_reflectivity(structure, self.q)
        label = self.quantize_labels(params)
        return refl, label

    # ------------------------------------------------
    # 3-4. 전체 데이터셋 생성
    # ------------------------------------------------
    def generate_dataset(self, n_samples: int, n_layers: int) -> tuple[np.ndarray, list[dict]]:
        data = []
        labels = []

        for _ in range(n_samples):
            params = self.sample_layer_parameters(n_layers=n_layers)
            refl, label = self.generate_sample(params)
            data.append(refl)
            labels.append(label)

        return np.array(data), labels



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


def main():
    from functools import partial
    wavelen: float = 1.54  # (Å)
    tth_range: tuple[float, float] = (0.01, 6.0)  # (1/Å)
    q_range: tuple[float, float] = map(partial(tth2q_wavelen, wavelen=wavelen), tth_range)  # (1/Å)
    q_points: int = 100
    thick_min, thick_max = 5.0, 2000.0    # Å
    sld_min, sld_max = 0.0, 140.0    # x1e-6 Å^-2
    sigma_min, sigma_max = 0.0, 10.0 # Å, 주로 0-10Å 세밀

    n_thick_coarse = 50    # coarse bins (log)
    n_sld_coarse = 40   # linear
    n_sig_coarse = 20   # linear

    # bins
    thick_bins = np.logspace(np.log10(thick_min), np.log10(thick_max), n_thick_coarse+1)
    sld_bins = np.linspace(sld_min, sld_max, n_sld_coarse+1)
    sig_bins = np.linspace(sigma_min, sigma_max, n_sig_coarse+1)
    xrr_generator = XRRDatasetGenerator(q_range, q_points, thick_bins, sld_bins, sig_bins)
    dataset = xrr_generator.generate_dataset(1, 2)

    print(dataset)


if __name__ == "__main__":
    main()
