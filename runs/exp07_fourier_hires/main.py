from pathlib import Path

import numpy as np
import simulate
import torch
from config import CONFIG, save_config
from dataset import XRR1LayerDataset
from evaluate import evaluate_pipeline
from torch.utils.data import DataLoader
from train import Trainer
from xrr_model import XRRPhysicsModel

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    print(f"✅ Seed set to {seed}")

def ensure_data_exists(qs, config, h5_path):
    if not h5_path.exists():
        print(f"📦 Data missing at {h5_path}. Generating clean source...")
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        simulate.generate_1layer_data(qs, config, h5_path)

def get_dataloaders(qs, config, h5_file, stats_file):
    t_cfg = config["training"]

    # 우리가 수정한 Physics-Augmentation 파라미터들
    common_args = {
        "qs": qs,
        "h5_file": h5_file,
        "stats_file": stats_file,
        "augment": t_cfg.get("augment", True),
        "expand_factor": t_cfg["expand_factor"],
        "aug_prob": t_cfg["aug_prob"],
        "intensity_scale": t_cfg["intensity_scale"],
        "q_shift_sigma": t_cfg["q_shift_sigma"],
        # [핵심] 77A 방지용 Resolution Smearing 범위 (물리적 q 단위)
        "res_sigma_range": (0.0001, 0.006)
    }

    loaders = []
    for mode in ["train", "val", "test"]:
        # shuffle은 학습 데이터에만 적용
        ds = XRR1LayerDataset(**common_args, mode=mode,
                            val_ratio=t_cfg["val_ratio"],
                            test_ratio=t_cfg["test_ratio"])

        loaders.append(DataLoader(
            ds,
            batch_size=t_cfg["batch_size"],
            shuffle=(mode=="train"),
            num_workers=t_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=(mode=="train") # Batch norm 안정성
        ))
    return loaders


def register_nan_hooks(model):
    """
    모델의 모든 레이어에 NaN/Inf 감지 훅을 설치합니다.
    문제가 발생하면 즉시 레이어 이름과 값을 출력하고 프로그램을 멈춥니다.
    """
    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"🚨 [NaN Detected] Forward Pass - Layer: {module}")
                raise RuntimeError(f"NaN found in output of {module}")
            if torch.isinf(output).any():
                print(f"⚠️ [Inf Detected] Forward Pass - Layer: {module}")
                # Inf는 즉시 에러는 아니지만 NaN의 전조증상임
                print(f"   - Max val: {output.max().item()}, Min val: {output.min().item()}")

    def backward_hook(module, grad_input, grad_output):
        # grad_output: 이 레이어에서 나가는 그라디언트
        if grad_output is not None:
            for i, grad in enumerate(grad_output):
                if isinstance(grad, torch.Tensor):
                    if torch.isnan(grad).any():
                        print(f"🚨 [NaN Detected] Backward Pass (Gradient) - Layer: {module}")
                        raise RuntimeError(f"NaN found in gradient of {module}")
                    if torch.isinf(grad).any():
                        print(f"⚠️ [Inf Detected] Backward Pass (Gradient) - Layer: {module}")

    print("🔎 Installing NaN hooks on all layers...")
    for name, module in model.named_modules():
        # 컨테이너(Sequential 등)가 아닌 실제 연산 레이어에만 등록
        if len(list(module.children())) == 0: 
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)

def register_debug_hooks(model):
    print("🕵️‍♀️ Installing Debug Hooks (Input/Weight Inspector)...")

    def forward_hook(module, input, output):
        # input은 튜플로 들어옵니다 (x, )
        x = input[0]

        # 1. 입력 데이터 검사
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"\n🚨 [CRITICAL] Input is dirty BEFORE entering {module}")
            print(f"   - Input min: {x.min().item()}, max: {x.max().item()}")
            print(f"   - Input NaNs: {torch.isnan(x).sum().item()}")
            raise RuntimeError(f"Bad Input at {module}")

        # 2. 가중치(Weights) 검사 (Conv/Linear 등)
        if hasattr(module, 'weight') and module.weight is not None:
            if torch.isnan(module.weight).any() or torch.isinf(module.weight).any():
                print(f"\n💀 [CRITICAL] Weights are ALREADY broken at {module}")
                print(f"   - Weight min: {module.weight.min().item()}, max: {module.weight.max().item()}")
                raise RuntimeError(f"Broken Weights at {module}")

        # 3. 출력 결과 검사 (여기가 터지면 연산 중 폭발)
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"\n💥 [CRITICAL] Output exploded AFTER {module}")
                print(f"   - Input stats: min={x.min().item():.2e}, max={x.max().item():.2e}")
                if hasattr(module, 'weight'):
                    print(f"   - Weight stats: min={module.weight.min().item():.2e}, max={module.weight.max().item():.2e}")
                raise RuntimeError(f"Explosion at {module}")

    for name, module in model.named_modules():
        # 컨테이너가 아닌 실제 연산 레이어에만 훅 등록
        if len(list(module.children())) == 0:
            module.register_forward_hook(forward_hook)

def main():
    print("🚀 EXP07 Launching: Physics-Informed Fourier Network")
    set_seed(42)
    # 1. 경로 및 설정 저장
    exp_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    h5_file = exp_dir / "dataset.h5"
    stats_file = exp_dir / "stats.pt"
    checkpoint_file = exp_dir / "best.pt"
    qs_file = exp_dir / "qs.npy"

    # 2. Grid 생성 (학습의 기준점)
    # [Tip] 재학습 시 Cubic Spline을 쓰므로 포인트를 2000개로 넉넉히 유지
    sim_cfg = CONFIG["simulation"]
    qs = np.linspace(sim_cfg["q_min"], sim_cfg["q_max"], sim_cfg["q_points"]).astype(np.float32)
    np.save(qs_file, qs)
    save_config(CONFIG, exp_dir / "config.json")
    # 3. 데이터 준비 및 로드
    ensure_data_exists(qs, CONFIG, h5_file)
    train_loader, val_loader, test_loader = get_dataloaders(qs, CONFIG, h5_file, stats_file)

    # 4. 모델 초기화
    print("🧠 Initializing Fourier Physics Network...")
    m_cfg = CONFIG["model"]
    model = XRRPhysicsModel(
        q_len=sim_cfg["q_points"],
        input_channels=2, # [LogR, Mask]
        output_dim=3,     # [Thick, Rough, SLD]
        n_channels=m_cfg["n_channels"],
        depth=m_cfg["depth"],
        mlp_hidden=m_cfg["mlp_hidden"],
        dropout=m_cfg["dropout"],
        use_fourier=m_cfg["use_fourier"],
        fourier_scale=m_cfg["fourier_scale"]
    )

    # Debug mode
    register_debug_hooks(model)

    # 5. Trainer 실행
    trainer = Trainer(
        model, train_loader, val_loader, exp_dir,
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        patience=CONFIG["training"]["patience"]
    )

    print("🔥 Training Start...")
    resume_path = exp_dir / "last.pt"
    trainer.train(
        CONFIG["training"]["epochs"],
        resume_from=resume_path if resume_path.exists() else None
    )

    # 6. 최종 평가 (Physics Report 생성)
    print("\n🏁 Running Final Physics-based Evaluation...")
    if checkpoint_file.exists():
        evaluate_pipeline(
            test_loader, checkpoint_file, stats_file, qs,
            report_img_path=exp_dir / "evaluation_report.png",
            report_csv_path=exp_dir / "evaluation_results.csv",
            report_history_path=exp_dir / "training_history.png"
        )


if __name__ == "__main__":
    main()

