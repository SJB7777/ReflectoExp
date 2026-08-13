from pathlib import Path

import numpy as np

from reflecto_exp.simulate.simul_genx import XRRSimulator


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
        "has_noise": False
    }
    # config의 param_ranges 키 -> XRRSimulator 인자명 매핑
    range_keys = {
        "thickness": "thickness_range",
        "roughness": "roughness_range",
        "sld": "sld_range",
        "sio2_thickness": "sio2_thick_range",
        "sio2_roughness": "sio2_rough_range",
        "sio2_sld": "sio2_sld_range",
        "sub_roughness": "sub_rough_range",
    }
    for cfg_key, arg_name in range_keys.items():
        value = param_ranges.get(cfg_key)
        if value is not None:
            simulator_args[arg_name] = tuple(value)

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

    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    h5_file = exp_dir / "dataset.h5"
    qs: np.ndarray = np.linspace(
        CONFIG["simulation"]["q_min"],
        CONFIG["simulation"]["q_max"],
        CONFIG["simulation"]["q_points"])
    generate_1layer_data(qs, CONFIG, h5_file)
