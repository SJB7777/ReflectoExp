import numpy as np

from reflecto_exp.physics_utils import tth2q
from reflecto_exp.simulate.simul_genx import ParamSet, XRRSimulator


def generate_dummy_xrr(qs, param: ParamSet):


    n_layers: int = 2
    n_samples: int = 1_000_000
    sim: XRRSimulator = XRRSimulator(qs, n_layers, n_samples)
    R = sim.simulate_one([param], has_noise=True)

    return R


if __name__ == '__main__':
    from pathlib import Path
        # q-space 생성 (당신 config와 동일)
    wavelen: float = 1.54  # (nm)
    tth_min: float = 1.0   # degree
    tth_max: float = 6.0
    q_min: float = tth2q(tth_min, wavelen)  # (1/Å)
    q_max: float = tth2q(tth_max, wavelen)
    print(q_min, q_max)
    q_n: int = 200
    qs: np.ndarray = np.linspace(q_min, q_max, q_n)
    param = ParamSet(42, 3.5, 5)
    R_dummy = generate_dummy_xrr(qs, param)
    save_dir = Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer\real_tester")
    np.save(save_dir / "measured_I.npy", R_dummy)
    np.save(save_dir / "measured_q.npy", qs)
