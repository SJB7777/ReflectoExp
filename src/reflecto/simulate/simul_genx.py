from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import h5py
import numpy as np
from genx.models.spec_nx import Layer, Sample, Specular, Stack
from numpy.typing import NDArray
from tqdm import tqdm

from reflecto.consts_genx import AIR, SUBSTRATE_SI, XRAY_TUBE, SURFACE_SIO2
from reflecto.simulate.noise import add_noise


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


@overload
def tth2q_wavelen(tth: float, wavelen: float = 1.54) -> float: ...

@overload
def tth2q_wavelen(tth: NDArray[np.float64], wavelen: float = 1.54) -> NDArray[np.float64]: ...

def tth2q_wavelen(tth, wavelen=1.54):
    """
    Convert 2θ (in degrees) and wavelength (in Å) to scattering vector q (in 1/Å).
    """
    th_rad = np.deg2rad(0.5 * tth)
    result = (4 * np.pi / wavelen) * np.sin(th_rad)

    if isinstance(tth, (int, float)):
        return float(result)
    return result


def build_sample(params: list[ParamSet]) -> Sample:
    """genx Sample를 연속 파라미터로부터 생성."""

    layers: list[Layer] = [SURFACE_SIO2]
    for param in params:
        layer = Layer(
            param.thickness,
            f = complex(param.sld, 0),
            dens=1,
            sigma=param.roughness
            )
        layers.append(layer)
    stack = Stack(Layers=layers, Repetitions=1)
    sample = Sample(
        Stacks=[stack],
        Ambient=AIR,
        Substrate=SUBSTRATE_SI
    )

    return sample

def calc_refl(sample: Sample, qs: np.ndarray) -> np.ndarray | None:
    """
    Compute reflectivity including beam footprint correction.

    Parameters
    ----------
    structure : refnx.reflect.Structure
        The interfacial structure.
    q : np.ndarray
        Momentum transfer values (Å⁻¹).

    Returns
    -------
    np.ndarray
        Reflectivity values corrected for beam footprint.
    """
    # 1. 기본 ReflectModel 계산
    reflectivity = Specular(qs, sample, XRAY_TUBE)
    return reflectivity if isinstance(reflectivity, np.ndarray) else None


def params2refl(params, qs):
    sample = build_sample(params)
    return calc_refl(sample, qs)

class XRRSimulator:
    def __init__(
            self,
            qs: np.ndarray,
            n_layers:int,
            n_samples: int,
            thickness_range: tuple[float, float] = (30.0, 100.0),
            roughness_range: tuple[float, float] = (0.0, 10.0),
            sld_range: tuple[float, float] = (0.5, 10.0),
            max_total_thickness: float = 250,
            has_noise: bool = True,
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

    def sample_thicknesses_uniform_with_limit(self):
        a, b = self.thick_range
        while True:
            t = np.random.uniform(a, b, size=self.n_layers)
            if t.sum() <= self.max_total_thickness:
                return t

    def make_params_refl(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
        """제약 조건을 만족하는 파라미터 생성 (Rejection-Free)"""
        for _ in range(0, self.n_samples):
            # 총 두께 제한을 자동으로 만족하는 두께 샘플링
            # thicknesses = self.sample_thicknesses_divide_and_conquer()
            thicknesses = self.sample_thicknesses_uniform_with_limit()
            # 거칠기 제약: 단일 층 두께의 3% 이하
            rough_min, rough_max = self.rough_range
            max_roughness_per_layer = np.minimum(thicknesses * 0.03, rough_max)
            roughnesses = np.random.uniform(rough_min, max_roughness_per_layer, self.n_layers)

            slds = np.random.uniform(*self.sld_range, self.n_layers)
            params: list[ParamSet] = []
            for t, r, s in zip(thicknesses, roughnesses, slds, strict=True):
                params.append(ParamSet(t, r, s))

            refl = self.simulate_one(params, has_noise=self.has_noise)

            yield thicknesses, roughnesses, slds, refl

    def simulate_one(self, params, has_noise=True) -> np.ndarray | None:
        structure = build_sample(params)
        refl = calc_refl(structure, self.qs)
        if refl is None:
            return None
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


def main() -> int | None:
    import matplotlib.pyplot as plt
    root: Path = Path(r"D:\03_Resources\Data\XRR_AI\data")
    root.mkdir(parents=True, exist_ok=True)
    file: Path = root / "xrr_data.h5"
    # Measurement Configurations
    wavelen: float = 1.54  # (nm)
    tth_min: float = 0.1   # degree
    tth_max: float = 6.0
    q_min: float = tth2q_wavelen(tth_min, wavelen)  # (1/Å)
    q_max: float = tth2q_wavelen(tth_max, wavelen)
    q_n: int = 200
    qs: np.ndarray = np.linspace(q_min, q_max, q_n)

    mean_thickness = 300
    mean_sld = 4.46
    thickness_range = (mean_thickness * 0.8, mean_thickness * 1.2)
    sld_range = (mean_sld * 0.8, mean_sld * 1.2)
    n_layers: int = 2
    n_samples: int = 1_000_000
    xrr_simulator: XRRSimulator = XRRSimulator(qs, n_layers, n_samples, has_noise=False, thickness_range=thickness_range, sld_range=sld_range)
    thicknesses, roughnesses, slds, refl = next(xrr_simulator.make_params_refl())
    print(thicknesses, roughnesses, slds)
    if refl is None:
        return 1
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(qs, refl)
    ax.set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
