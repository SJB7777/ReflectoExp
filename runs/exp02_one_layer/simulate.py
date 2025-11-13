import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

from reflecto.simulator.simulator import XRRSimulator, tth2q_wavelen

# ==================== 설정 ====================
OUTPUT_DIR = Path(r"D:\03_Resources\Data\XRR_AI\data\one_layer")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1-layer 전용 설정
MEASUREMENT_CONFIG = {
    "wavelength": 1.54,  # Å
    "tth_min": 1.0,      # degree
    "tth_max": 6.0,
    "q_points": 200,
}

PARAMETER_RANGES = {
    "thickness": (5.0, 200.0),   # nm
    "roughness": (0.0, 10.0),    # Å
    "sld": (0.0, 140.0),         # 1e-6 Å^-2
}

N_SAMPLES = 3_000_000
# ============================================

def generate_1layer_data():
    """1-layer XRD 데이터 생성"""
    print("=== 1-Layer XRR 데이터 생성 시작 ===")

    # q 벡터 생성
    q_min = tth2q_wavelen(MEASUREMENT_CONFIG["tth_min"], MEASUREMENT_CONFIG["wavelength"])
    q_max = tth2q_wavelen(MEASUREMENT_CONFIG["tth_max"], MEASUREMENT_CONFIG["wavelength"])
    q_values = np.linspace(q_min, q_max, MEASUREMENT_CONFIG["q_points"])

    # 시뮬레이터 초기화 (n_layers=1)
    simulator = XRRSimulator(
        qs=q_values,
        n_layers=1,  # ** 1-layer 전용 **
        n_samples=N_SAMPLES,
        thickness_range=PARAMETER_RANGES["thickness"],
        roughness_range=PARAMETER_RANGES["roughness"],
        sld_range=PARAMETER_RANGES["sld"],
    )

    # HDF5 파일 생성
    output_path = OUTPUT_DIR / "xrr_1layer_small.h5"
    simulator.save_hdf5(output_path, show_progress=True)

    print(f"데이터 저장 완료: {output_path}")
    print(f"   - 샘플 수: {N_SAMPLES}")
    print(f"   - q 포인트: {len(q_values)}")
    print(f"   - 파라미터 범위: {PARAMETER_RANGES}")

    return output_path

if __name__ == "__main__":
    generate_1layer_data()
