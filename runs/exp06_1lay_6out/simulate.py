from pathlib import Path

import numpy as np

from reflecto.simulate.simul_genx import XRRSimulator


def generate_1layer_data(qs: np.ndarray, config: dict, h5_file: Path | str):
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

    simulator_args: dict = {
        "qs": qs,
        "n_layers": 1,
        "n_samples": simulation["n_samples"],
        "has_noise": True
    }
    if param_ranges["thickness"] is not None:
        simulator_args["thickness_range"] = param_ranges["thickness"]
    if param_ranges["roughness"] is not None:
        simulator_args["roughness_range"] = param_ranges["roughness"]
    if param_ranges["sld"] is not None:
        simulator_args["sld_range"] = param_ranges["sld"]
    if param_ranges["sio2_thickness"] is not None:
        simulator_args["sio2_thickness_range"] = param_ranges["sio2_thickness"]
    if param_ranges["sio2_roughness"] is not None:
        simulator_args["sio2_roughness_range"] = param_ranges["sio2_roughness"]
    if param_ranges["sio2_sld"] is not None:
        simulator_args["sio2_sld_range"] = param_ranges["sio2_sld"]
    simulator = XRRSimulator(
        **simulator_args
    )

    simulator.save_hdf5(h5_file, show_progress=True)

    print(f"\n 데이터 저장 완료: {h5_file}")
    print(f"   - 샘플 수: {simulation['n_samples']:,}")
    print(f"   - q 포인트: {len(qs)}")
    print(f"   - 파라미터 범위: {param_ranges}")


if __name__ == "__main__":
    from config import CONFIG

    from reflecto.math_utils import powerspace

    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    h5_file = exp_dir / "dataset.h5"
    qs: np.ndarray = powerspace(
        CONFIG["simulation"]["q_min"],
        CONFIG["simulation"]["q_max"],
        CONFIG["simulation"]["q_points"],
        CONFIG["simulation"]["power"])
    generate_1layer_data(qs, CONFIG, h5_file)
