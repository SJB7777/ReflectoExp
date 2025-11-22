from pathlib import Path

import numpy as np

from reflecto.physics_utils import tth2q
from reflecto.simulate.simulator import XRRSimulator


def generate_1layer_data(config: dict, h5_file: Path | str):
    """
    1-layer XRR 데이터 생성

    Args:
        config: main.py의 CONFIG 딕셔너리 (simulation, paths, param_ranges 포함)
    """
    print("=== 1-Layer XRR 데이터 생성 시작 ===")

    # config에서 모든 파라미터 추출
    simulation = config["simulation"]
    param_ranges = config["param_ranges"]
    h5_file = Path(h5_file)
    # 출력 디렉토리 생성
    output_dir = h5_file.parent
    output_dir.mkdir(exist_ok=True, parents=True)

    # q 벡터 생성
    q_min = tth2q(simulation["tth_min"], simulation["wavelength"])
    q_max = tth2q(simulation["tth_max"], simulation["wavelength"])
    q_values = np.linspace(q_min, q_max, simulation["q_points"])

    # 시뮬레이터 초기화 (n_layers=1)
    simulator = XRRSimulator(
        qs=q_values,
        n_layers=1,
        n_samples=simulation["n_samples"],
        thickness_range=param_ranges["thickness"],
        roughness_range=param_ranges["roughness"],
        sld_range=param_ranges["sld"],
        has_footprint=True
    )


    simulator.save_hdf5(h5_file, show_progress=True)

    print(f"\n 데이터 저장 완료: {h5_file}")
    print(f"   - 샘플 수: {simulation['n_samples']:,}")
    print(f"   - q 포인트: {len(q_values)}")
    print(f"   - 파라미터 범위: {param_ranges}")

if __name__ == "__main__":
    pass
