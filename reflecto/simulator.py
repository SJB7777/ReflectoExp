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
    return model(q) # nm -> Å

def xrr_simulate(qs, params: list[ParamSet], /, has_noise=False) -> np.ndarray:
    structure = build_structure(params)
    refl = compute_reflectivity(structure, qs)

    if has_noise:
        # TODO: add noise
        pass

    return refl


def main() -> None:
    import random

    import matplotlib.pyplot as plt


    # Material Configurations
    thick_min, thick_max = 5.0, 200.0    # nm
    sld_min, sld_max = 0.0, 140.0    # x1e-6 Å^-2
    rough_min, rough_max = 0.0, 10.0  # Å, 주로 0-10Å 세밀
    n_layer: int = 2
    params: list[ParamSet] = []
    for _ in range(n_layer):
        thickness = random.uniform(thick_min, thick_max + 1e-9)
        roughness = random.uniform(rough_min, rough_max + 1e-9)
        sld = random.uniform(sld_min, sld_max + 1e-9)
        params.append(ParamSet(thickness, roughness, sld))

    # Measurement Configurations
    wavelen: float = 1.54  # (nm)
    tth_min: float = 1.0   # degree
    tth_max: float = 6.0
    q_min: float = tth2q_wavelen(tth_min, wavelen)  # (1/Å)
    q_max: float = tth2q_wavelen(tth_max, wavelen)
    q_n: int = 100
    qs: np.ndarray = np.linspace(q_min, q_max, q_n)

    # Data Simuation
    reflectivity = xrr_simulate(qs, params, has_noise=True)

    # Plot graph
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Simulated XRR")
    ax.plot(qs, reflectivity)
    ax.set_yscale("log")
    ax.set_xlabel("Q_z [1/Å]")
    ax.set_ylabel("Intensity [arb. unit]")

    plt.show()


if __name__ == "__main__":
    main()
