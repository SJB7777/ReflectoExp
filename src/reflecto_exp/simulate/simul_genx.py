from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from genx.models.spec_nx import Layer, Sample, Specular, Stack
from tqdm import tqdm

from reflecto_exp.consts_genx import AIR, SUBSTRATE_SI, SURFACE_SIO2, XRAY_TUBE
from reflecto_exp.physics_utils import r_e
from reflecto_exp.simulate.noise import add_noise


@dataclass
class ParamSet:
    thickness: float
    roughness: float
    sld: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.thickness, self.roughness, self.sld], dtype=np.float64)

    @classmethod
    def from_genx_layer(cls, layer: Layer) -> "ParamSet":
        """
        GenX Layer 객체를 ParamSet으로 변환합니다.
        f (scattering factor)를 SLD (10^-6 A^-2) 단위로 역변환합니다.
        Formula: SLD_val = (f.real * dens * r_e) * 1e6
        """
        # f는 복소수이므로 실수부(real)만 취함 (전자밀도 관련)
        sld_val = (layer.f.real * layer.dens * r_e) * 1e6

        return cls(
            thickness=float(layer.d),
            roughness=float(layer.sigma),
            sld=float(sld_val)
        )

    def to_layer(self) -> Layer:
        """Return genx Layer object"""
        return Layer(
        d=self.thickness,
        f=get_f(self.sld),
        dens=1.0,
        sigma=self.roughness
        )

    def __format__(self, format_spec: str) -> str:
        formatted = [
            f"{name}={getattr(self, name):{format_spec}}"
            for name in self.__dataclass_fields__.keys()
        ]
        return f"{self.__class__.__name__}({', '.join(formatted)})"


def get_f(sld_1e6: float) -> complex:
    """
    Convert SLD (10^-6 A^-2) to Scattering Factor (f).
    Assuming density = 1.0 for GenX calculation.
    Formula: f = SLD / (rho * r_e)
    """
    return complex((sld_1e6 * 1e-6) / r_e, 0)


def build_sample(film_params: list[ParamSet], sio2_param: ParamSet, sub_param: ParamSet | None = None) -> Sample:
    """
    Create GenX Sample with explicit Si Substrate.
    Structure: Si Substrate -> Film -> Surface SiO2 -> Ambient
    """
    if sub_param is None:
        sub_param = ParamSet(0, 3, 20.1)
    layers = [sio2_param.to_layer()]
    for param in film_params:
        layers.append(
            param.to_layer()
        )

    stack = Stack(Layers=layers, Repetitions=1)

    sample = Sample(
        Stacks=[stack],
        Ambient=AIR,
        Substrate=sub_param.to_layer()
    )
    return sample


def calc_refl(sample: Sample, qs: np.ndarray) -> np.ndarray:
    """
    Compute reflectivity.
    Raises RuntimeError if simulation fails.
    """
    reflectivity = Specular(qs, sample, XRAY_TUBE)

    if not isinstance(reflectivity, np.ndarray):
        raise RuntimeError(f"GenX Simulation Failed. Return value: {reflectivity}")

    return reflectivity


def param2refl(qs: np.ndarray, film_params: list[ParamSet], sio2_param: ParamSet | None = None, sub_param: ParamSet | None = None) -> np.ndarray:
    if sio2_param is None:
        sio2_param = ParamSet.from_genx_layer(SURFACE_SIO2)
    if sub_param is None:
        sub_param = ParamSet.from_genx_layer(SUBSTRATE_SI)
    sample = build_sample(film_params, sio2_param, sub_param)
    return calc_refl(sample, qs)


class XRRSimulator:
    def __init__(
            self,
            qs: np.ndarray,
            n_samples: int,
            n_layers: int,
            # Ranges
            thickness_range: tuple[float, float] = (10.0, 500.0),
            roughness_range: tuple[float, float] = (0.0, 20.0),
            sld_range: tuple[float, float] = (5.0, 150.0),
            sio2_thick_range: tuple[float, float] = (10.0, 25.0),
            sio2_rough_range: tuple[float, float] = (2.0, 5.0),
            sio2_sld_range: tuple[float, float] = (5, 22),
            sub_rough_range: tuple[float, float] = (0.0, 5.0),
            has_noise: bool = False,
            ):

        self.qs = qs
        self.n_samples = n_samples
        self.n_layers = n_layers
        self.t_min, self.t_max = thickness_range
        self.r_min, self.r_max = roughness_range
        self.s_min, self.s_max = sld_range

        self.st_min, self.st_max = sio2_thick_range
        self.sr_min, self.sr_max = sio2_rough_range
        self.ss_min, self.ss_max = sio2_sld_range

        self.sub_r_min, self.sub_r_max = sub_rough_range

        self.has_noise = has_noise

    def generate_batch(self) -> Iterator[tuple[list[ParamSet], ParamSet, np.ndarray]]:
        """Generator for (Film_Params, SiO2_Params, Reflectivity)"""

        for _ in range(self.n_samples):
            # 1. Main Film
            film_params: list[ParamSet] = []
            for _ in range(self.n_layers):
                f_d = np.random.uniform(self.t_min, self.t_max)
                max_r = min(self.r_max, f_d * 0.3)
                f_sig = np.random.uniform(self.r_min, max_r)
                f_sld = np.random.uniform(self.s_min, self.s_max)
                film_params.append(ParamSet(f_d, f_sig, f_sld))

            # 2. SiO2
            s_d = np.random.uniform(self.st_min, self.st_max)
            s_sig = np.random.uniform(self.sr_min, self.sr_max)
            s_sld = np.random.uniform(self.ss_min, self.ss_max)
            sio2_p = ParamSet(s_d, s_sig, s_sld)

            sub_sig = np.random.uniform(self.sub_r_min, self.sub_r_max)
            sub_param = ParamSet.from_genx_layer(SUBSTRATE_SI)
            sub_param.roughness = sub_sig
            # 3. Simulate
            try:
                refl = param2refl(self.qs, film_params, sio2_p, sub_param)
            except RuntimeError as e:
                print(f"[Simulation Error] Skipping sample: {e}")
                raise e

            if self.has_noise:
                refl = add_noise(refl)

            yield film_params, sio2_p, refl

    def save_hdf5(self, file: str | Path, show_progress: bool = True) -> None:
        file = Path(file)

        with h5py.File(file, 'w') as hf:
            hf.create_dataset("q", data=self.qs)

            d_thick = hf.create_dataset("thickness", (self.n_samples, self.n_layers), dtype='f4')
            d_rough = hf.create_dataset("roughness", (self.n_samples, self.n_layers), dtype='f4')
            d_sld = hf.create_dataset("sld", (self.n_samples, self.n_layers), dtype='f4')

            d_s_thick = hf.create_dataset("sio2_thickness", (self.n_samples, 1), dtype='f4')
            d_s_rough = hf.create_dataset("sio2_roughness", (self.n_samples, 1), dtype='f4')
            d_s_sld = hf.create_dataset("sio2_sld", (self.n_samples, 1), dtype='f4')

            d_refl = hf.create_dataset("R", (self.n_samples, len(self.qs)), dtype='f4')

            iterator = self.generate_batch()
            if show_progress:
                iterator = tqdm(iterator, total=self.n_samples, desc="Simulating", dynamic_ncols=True)

            for i, (fps, sp, r) in enumerate(iterator):
                d_thick[i] = np.array([fp.thickness for fp in fps])
                d_rough[i] = np.array([fp.roughness for fp in fps])
                d_sld[i] = np.array([fp.sld for fp in fps])

                d_s_thick[i] = sp.thickness
                d_s_rough[i] = sp.roughness
                d_s_sld[i]   = sp.sld

                d_refl[i] = r


def main():
    # Test Execution
    import matplotlib.pyplot as plt

    qs = np.linspace(0.0, 0.5, 200)

    sim = XRRSimulator(
        qs=qs,
        n_samples=5,
        n_layers=2,
        thickness_range=(50, 150),
        roughness_range=(0, 5),
        sld_range=(8, 12),
        has_noise=False
    )

    print("Testing Simulation...")
    for films, sio2, refl in sim.generate_batch():
        print(f"Films: {films}")
        print(f"SiO2: {sio2}")

        plt.figure(figsize=(6,4))
        plt.plot(qs, refl)
        plt.yscale('log')
        title: str = f"SiO2: {sio2:.2g}\n"
        title += "\n".join(f"{i}: {film:.2g}" for i, film in enumerate(films, 1))
        plt.title(title)
        plt.tight_layout()
        plt.show()
        break


if __name__ == "__main__":
    main()
