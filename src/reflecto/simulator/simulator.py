from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from refnx.reflect import SLD, ReflectModel
from refnx.reflect.structure import Stack, Structure
from tqdm import tqdm

from reflecto.simulator.noise import add_noise


@dataclass
class ParamSet:
    thickness: float
    roughness: float
    sld: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.thickness, self.roughness, self.sld], dtype=np.float64)

    def __format__(self, format_spec: str) -> str:
        """클래스 전체에 포매팅 명령어 적용 (.3f, .2f 등)"""
        formatted = [
            f"{name}={getattr(self, name):{format_spec}}"
            for name in self.__dataclass_fields__.keys()
        ]
        return f"{self.__class__.__name__}({', '.join(formatted)})"


def tth2q_wavelen[T: (float, np.ndarray)](tth: T, wavelen: float = 1.54) -> T:
    """
    Convert 2θ (in degrees) and wavelength (in Å) to scattering vector q (in 1/Å).
    tth: degree
    wavelen: Å
    default Kalpha = 1.54

    -> q: 1/Å
    """
    if 0 > min(tth) or max(tth) > 90:
        raise ValueError(f"tth should have value under (0, 90) not {tth}")
    th_rad = np.radians(tth / 2.0)
    return (4 * np.pi / wavelen) * np.sin(th_rad)


def build_structure(params) -> Structure:
    """refnx Structure를 연속 파라미터로부터 생성."""
    air = SLD(0.0, name="Air")
    substrate = SLD(19.2, name="Substrate")

    stack = Stack(name="MultiLayer", repeats=len(params))

    for param in params:
        mat = SLD(param.sld, name=f"SLD={param.sld}")
        stack.append(mat(param.thickness, param.roughness))

    return air(0, 0) | stack | substrate(0, 3)

def compute_reflectivity(structure: Structure, q: np.ndarray) -> np.ndarray:
    model = ReflectModel(structure)
    return model(q) # nm -> Å

def apply_poisson_noise(arr: np.ndarray, s: float) -> np.ndarray:
    """Apply Poisson noise to an array."""
    expected_counts = s * arr
    noisy_counts = np.random.poisson(expected_counts)
    return noisy_counts / s


def get_background_noise(count: int, b_min: float, b_max: float) -> np.ndarray:
    """Generate background noise."""
    b = pow(10, np.random.uniform(b_min, b_max))
    return np.random.normal(b, 0.1 * b, count)

def add_noise(R):
    N = len(R)
    R_poisson = apply_poisson_noise(R, s=10 ** 7)
    uniform_noise = 1 + np.random.uniform(-0.01, 0.01, N)
    background_noise = get_background_noise(N, -8, -7)
    curve_scaling = np.random.uniform(0.99, 1.01)
    return R_poisson * uniform_noise * curve_scaling + background_noise

class XRRSimulator:
    def __init__(
            self,
            qs: np.ndarray,
            n_layers:int,
            n_samples: int,
            thickness_range: tuple[float, float] = (30.0, 100.0),
            roughness_range: tuple[float, float] = (0.0, 10.0),
            sld_range: tuple[float, float] = (0.5, 20.0),
            max_total_thickness: float = 250,
            has_noise: bool = True
            ):
        self.qs = qs
        self.n_layers = n_layers
        self.n_samples = n_samples

        self.thick_range = thickness_range    # Å
        self.rough_range = roughness_range  # Å, 주로 0-10Å 세밀
        self.sld_range = sld_range    # x1e-6 Å^-2
        self.max_total_thickness = max_total_thickness

        self.has_noise = has_noise

    def sample_thicknesses_divide_and_conquer(self) -> np.ndarray:
        """Dirichlet 분포를 사용하여 균일한 두께 샘플링"""
        min_total = self.n_layers * self.thick_range[0]
        total_thickness = np.random.uniform(min_total, self.max_total_thickness)

        proportions = np.random.dirichlet(np.ones(self.n_layers))
        thicknesses = total_thickness * proportions

        # 안전장치: 범위를 벗어나는 경우 클리핑
        thicknesses = np.clip(thicknesses, *self.thick_range)

        return thicknesses

    def make_params_refl(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """제약 조건을 만족하는 파라미터 생성 (Rejection-Free)"""
        for _ in range(0, self.n_samples):
            # 총 두께 제한을 자동으로 만족하는 두께 샘플링
            thicknesses = self.sample_thicknesses_divide_and_conquer()

            # 거칠기 제약: 단일 층 두께의 3% 이하
            rough_min, rough_max = self.rough_range
            max_roughness_per_layer = np.minimum(thicknesses * 0.03, rough_max)
            roughnesses = np.random.uniform(rough_min, max_roughness_per_layer, self.n_layers)

            slds = np.random.uniform(*self.sld_range, self.n_layers)
            params: list[ParamSet] = []
            for t, r, s in zip(thicknesses, roughnesses, slds, strict=True):
                params.append(ParamSet(t, r, s))

            refl = self.simulate_one(params)
            if self.has_noise:
                refl = add_noise(refl)
            yield thicknesses, roughnesses, slds, refl


    def simulate_one(self, params, has_noise=False) -> np.ndarray:
        structure = build_structure(params)
        refl = compute_reflectivity(structure, self.qs)

        self.refl = add_noise(refl) if has_noise else refl
        return self.refl

    def save_hdf5(self, file: str | Path, show_progress: bool = True) -> None:
        file = Path(file)

        with h5py.File(file, 'w') as hf:
            hf.create_dataset("q", data=self.qs)

            d_thick = hf.create_dataset("thickness", (self.n_samples, self.n_layers), dtype='f4')
            d_rough = hf.create_dataset("roughness", (self.n_samples, self.n_layers), dtype='f4')
            d_sld = hf.create_dataset("sld", (self.n_samples, self.n_layers), dtype='f4')
            d_refl = hf.create_dataset(
                "R", (self.n_samples, len(self.qs)), dtype='f4'
                )

            iterator = self.make_params_refl()
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total= self.n_samples,
                    desc="Saving HDF5",
                    dynamic_ncols=True,
                )

            for i, (thicknesses, roughnesses, slds, refl) in enumerate(iterator):

                d_thick[i] = thicknesses
                d_rough[i] = roughnesses
                d_sld[i] = slds
                d_refl[i] = refl


def main() -> None:
    root: Path = Path(r"D:\03_Resources\Data\XRR_AI\data")
    root.mkdir(parents=True, exist_ok=True)
    file: Path = root / "xrr_data.h5"
    # Measurement Configurations
    wavelen: float = 1.54  # (nm)
    tth_min: float = 1.0   # degree
    tth_max: float = 6.0
    q_min: float = tth2q_wavelen(tth_min, wavelen)  # (1/Å)
    q_max: float = tth2q_wavelen(tth_max, wavelen)
    q_n: int = 100
    qs: np.ndarray = np.linspace(q_min, q_max, q_n)

    n_layers: int = 2
    n_samples: int = 1_000_000
    xrr_simulator: XRRSimulator = XRRSimulator(qs, n_layers, n_samples)
    xrr_simulator.save_hdf5(file)


if __name__ == "__main__":
    main()
