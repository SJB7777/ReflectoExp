from pathlib import Path

from fitting_engine import GenXFitter

from reflecto.io_utils import load_xrr_dat
from reflecto.simulate.simul_genx import tth2q_wavelen
from runs.exp05_1layer_mask.inference import XRRInferenceEngine


def main():
    # =========================================================
    # 1. 설정 (분석할 데이터 경로 지정)
    # =========================================================
    # 분석하고 싶은 데이터 파일 경로를 입력하세요
    target_data_path = Path(r"D:\03_Resources\Data\XRR_AI\XRR_data\sample_test.dat")

    # NN 모델 가중치가 있는 폴더 (import 경로에 맞춰 설정)
    nn_weights_dir = Path("runs/exp05_1layer_mask")

    print("=== XRR Analysis Pipeline Started ===")
    print(f"Target File: {target_data_path.name}")

    # =========================================================
    # 2. 데이터 로드 (Data Loading)
    # =========================================================
    if not target_data_path.exists():
        print(f"[Error] 데이터 파일을 찾을 수 없습니다: {target_data_path}")
        return

    # load_xrr_dat 함수가 (q, R)을 반환한다고 가정
    # 만약 tth를 반환한다면 변환 로직이 필요합니다.
    # 여기서는 이미 q로 변환되어 나온다고 가정하거나, 내부에서 변환한다고 봅니다.
    # (일반적으로: q = 4 * pi * sin(theta) / lambda)
    tths, R_raw = load_xrr_dat(target_data_path)
    q_raw = tth2q_wavelen(tths)
    # 데이터 전처리 (유효 구간 필터링 등 필요시 추가)
    # 예: q > 0.02 구간만 사용
    mask = q_raw > 0.02
    q_raw = q_raw[mask]
    R_raw = R_raw[mask]

    print(f"[Data] Loaded {len(q_raw)} points.")

    # =========================================================
    # 3. NN 초기값 예측 (Neural Network Inference)
    # =========================================================
    print("\n[Step 1] Neural Network Inference...")

    # 추론 엔진 초기화
    inference_engine = XRRInferenceEngine(exp_dir=nn_weights_dir)

    # 예측 실행 (thickness, roughness, sld 반환)
    pred_d, pred_sig, pred_sld = inference_engine.predict(q_raw, R_raw)

    print("   >>> NN Prediction:")
    print(f"       Thickness : {pred_d:.2f} Å")
    print(f"       Roughness : {pred_sig:.2f} Å")
    print(f"       SLD       : {pred_sld:.3f} (10⁻⁶ Å⁻²)")

    # GenXFitter에 넘겨줄 딕셔너리 생성
    nn_initial_params = {
        'd': pred_d,
        'sigma': pred_sig,
        'sld': pred_sld
    }

    # =========================================================
    # 4. GenX 정밀 피팅 (GenX Refinement)
    # =========================================================
    print("\n[Step 2] GenX Fitting (Optimization)...")

    # Fitter 초기화 (데이터 + 초기값 주입)
    fitter = GenXFitter(q_raw, R_raw, nn_initial_params)

    # 피팅 실행 (verbose=True로 과정 출력)
    final_results = fitter.run(verbose=True)

    # =========================================================
    # 5. 결과 리포트 및 시각화
    # =========================================================
    print("\n" + "="*40)
    print("FINAL ANALYSIS RESULT")
    print("="*40)

    # 결과 딕셔너리 예쁘게 출력
    for param_name, value in final_results.items():
        print(f"{param_name:15s}: {value:.4f}")
    print("="*40)

    # 그래프 그리기
    fitter.plot()


if __name__ == "__main__":
    main()
