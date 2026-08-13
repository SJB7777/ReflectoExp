"""
exp07 요소별 기여도 분해(Ablation) 러너.

exp02와 exp07은 시뮬레이터/q-grid/층구조/라벨 범위가 모두 달라 수치 직접 비교가
성립하지 않는다. 따라서 동일 데이터셋(exp07 h5) 위에서 개선 요소를 하나씩 끄고
학습하여 각 요소의 기여도를 분리 측정한다.

측정 요소:
  - Fourier feature   (model.use_fourier)
  - Physics augmentation (training.augment)
  - q 해상도          (simulation.q_points, 원본 h5에서 재샘플링)

주의: q_points를 줄여도 데이터 재생성은 불필요하다. XRRPreprocessor가 원본 q 그리드에서
target_q로 보간하므로, 2000점 h5 하나로 저해상도 조건을 만들 수 있다.

사용:
    python runs/exp07_fourier_hires/ablation.py --dry-run
    python runs/exp07_fourier_hires/ablation.py
    python runs/exp07_fourier_hires/ablation.py --only A_full C_no_augment --epochs 5
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from config import CONFIG, save_config
from dataset import XRR1LayerDataset
from evaluate import evaluate_pipeline
from torch.utils.data import DataLoader
from train import Trainer
from xrr_model import XRRPhysicsModel

import simulate

# -----------------------------------------------------------------------------
# Variant 정의: 기준(CONFIG)에 덮어쓸 부분만 기술
# -----------------------------------------------------------------------------
VARIANTS: dict[str, dict] = {
    "A_full": {},
    "B_no_fourier": {
        "model": {"use_fourier": False},
    },
    "C_no_augment": {
        "training": {"augment": False, "expand_factor": 1},
    },
    "D_lowres_200": {
        "simulation": {"q_points": 200},
        "model": {"depth": 4},
    },
    "E_exp02_like": {
        "simulation": {"q_points": 200},
        "model": {"depth": 4, "use_fourier": False},
        "training": {"augment": False, "expand_factor": 1},
    },
}

PARAM_NAMES = ["Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)"]


def deep_merge(base: dict, override: dict) -> dict:
    """base의 깊은 복사본에 override를 재귀 병합"""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def build_qs(sim_cfg: dict) -> np.ndarray:
    return np.linspace(sim_cfg["q_min"], sim_cfg["q_max"], sim_cfg["q_points"]).astype(np.float32)


def make_loaders(qs: np.ndarray, cfg: dict, h5_file: Path, stats_file: Path):
    t_cfg = cfg["training"]
    common = {
        "qs": qs,
        "h5_file": h5_file,
        "stats_file": stats_file,
        "augment": t_cfg.get("augment", True),
        "expand_factor": t_cfg["expand_factor"],
        "aug_prob": t_cfg["aug_prob"],
        "intensity_scale": t_cfg["intensity_scale"],
        "q_shift_sigma": t_cfg["q_shift_sigma"],
        "res_sigma_range": tuple(t_cfg.get("res_sigma_range", (0.0001, 0.006))),
        "augment_eval": t_cfg.get("augment_eval", False),
    }

    loaders = []
    for mode in ["train", "val", "test"]:
        ds = XRR1LayerDataset(
            **common, mode=mode,
            val_ratio=t_cfg["val_ratio"], test_ratio=t_cfg["test_ratio"]
        )
        loaders.append(DataLoader(
            ds,
            batch_size=t_cfg["batch_size"],
            shuffle=(mode == "train"),
            num_workers=t_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=(mode == "train"),
        ))
    return loaders


def build_model(cfg: dict, q_len: int) -> XRRPhysicsModel:
    m = cfg["model"]
    return XRRPhysicsModel(
        q_len=q_len,
        input_channels=2,
        output_dim=3,
        n_channels=m["n_channels"],
        depth=m["depth"],
        mlp_hidden=m["mlp_hidden"],
        dropout=m["dropout"],
        use_fourier=m["use_fourier"],
        fourier_scale=m["fourier_scale"],
    )


def steps_per_epoch(loader: DataLoader) -> int:
    return len(loader)


def resolve_epochs(cfg: dict, train_loader: DataLoader, budget_steps: int | None,
                   epochs_override: int | None) -> int:
    """
    Variant마다 epoch당 step 수가 다르므로(augmentation expand_factor 차이),
    epoch이 아닌 optimizer step 수를 맞춰야 공정 비교가 된다.
    """
    if epochs_override is not None:
        return epochs_override
    if budget_steps is None:
        return cfg["training"]["epochs"]

    spe = max(1, steps_per_epoch(train_loader))
    return max(1, math.ceil(budget_steps / spe))


def run_variant(name: str, cfg: dict, root_dir: Path, h5_file: Path,
                budget_steps: int | None, epochs_override: int | None) -> dict:
    print("\n" + "=" * 80)
    print(f"[Ablation] Variant: {name}")
    print("=" * 80)

    set_seed(42)

    var_dir = root_dir / "ablation" / name
    var_dir.mkdir(parents=True, exist_ok=True)
    stats_file = var_dir / "stats.pt"
    checkpoint_file = var_dir / "best.pt"

    qs = build_qs(cfg["simulation"])
    np.save(var_dir / "qs.npy", qs)
    save_config(cfg, var_dir / "config.json")

    train_loader, val_loader, test_loader = make_loaders(qs, cfg, h5_file, stats_file)
    epochs = resolve_epochs(cfg, train_loader, budget_steps, epochs_override)

    model = build_model(cfg, q_len=len(qs))
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  q_points      : {len(qs)}  (Δq={qs[1] - qs[0]:.3e})")
    print(f"  use_fourier   : {cfg['model']['use_fourier']}")
    print(f"  depth         : {cfg['model']['depth']}")
    print(f"  augment       : {cfg['training']['augment']} (expand ×{cfg['training']['expand_factor']})")
    print(f"  params        : {n_params:,}")
    print(f"  train samples : {len(train_loader.dataset):,}")
    print(f"  steps/epoch   : {steps_per_epoch(train_loader):,}")
    print(f"  epochs        : {epochs}  → total steps ≈ {epochs * steps_per_epoch(train_loader):,}")

    trainer = Trainer(
        model, train_loader, val_loader, var_dir,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
    )
    resume_path = var_dir / "last.pt"
    trainer.train(epochs, resume_from=resume_path if resume_path.exists() else None)

    metrics = evaluate_pipeline(
        test_loader, checkpoint_file, stats_file, qs,
        report_img_path=var_dir / "evaluation_correlation.png",
        report_csv_path=var_dir / "evaluation_results.csv",
        report_history_path=var_dir / "training_history.png",
    )

    row: dict = {
        "variant": name,
        "use_fourier": cfg["model"]["use_fourier"],
        "depth": cfg["model"]["depth"],
        "q_points": len(qs),
        "augment": cfg["training"]["augment"],
        "expand_factor": cfg["training"]["expand_factor"],
        "params": n_params,
        "train_samples": len(train_loader.dataset),
        "epochs_run": epochs,
        "total_steps": epochs * steps_per_epoch(train_loader),
        "best_val_loss": trainer.best_val_loss,
    }

    if metrics is not None:
        for i, pname in enumerate(PARAM_NAMES):
            clean = pname.split(" (")[0]
            row[f"{clean}_MAE"] = float(metrics["mae"][i])
            row[f"{clean}_RMSE"] = float(metrics["rmse"][i])
            row[f"{clean}_MAPE"] = float(metrics["mape"][i])
            row[f"{clean}_R2"] = float(metrics["r2"][i])
    else:
        print("⚠️  Evaluation skipped (missing checkpoint or stats).")

    return row


def print_plan(cfgs: dict[str, dict]):
    print("\n" + "=" * 100)
    print(f"{'Variant':<16} | {'q_pts':>6} | {'depth':>5} | {'fourier':>7} | {'augment':>7} | {'expand':>6} | {'params':>10}")
    print("-" * 100)
    for name, cfg in cfgs.items():
        qn = cfg["simulation"]["q_points"]
        model = build_model(cfg, q_len=qn)
        n = sum(p.numel() for p in model.parameters())
        print(f"{name:<16} | {qn:>6} | {cfg['model']['depth']:>5} | "
              f"{str(cfg['model']['use_fourier']):>7} | {str(cfg['training']['augment']):>7} | "
              f"{cfg['training']['expand_factor']:>6} | {n:>10,}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="exp07 ablation runner")
    parser.add_argument("--only", nargs="+", default=None,
                        help=f"실행할 variant 이름 (기본: 전체). 선택지: {list(VARIANTS)}")
    parser.add_argument("--epochs", type=int, default=None,
                        help="모든 variant의 epoch를 강제 지정 (스모크 테스트용)")
    parser.add_argument("--budget-mode", choices=["steps", "epochs"], default="steps",
                        help="steps: 기준 variant의 총 optimizer step 수에 맞춰 epoch 자동 조정 (기본). "
                             "epochs: config의 epochs를 그대로 사용")
    parser.add_argument("--baseline", default="A_full",
                        help="step 예산 기준이 되는 variant")
    parser.add_argument("--dry-run", action="store_true",
                        help="학습 없이 실행 계획만 출력")
    args = parser.parse_args()

    names = args.only if args.only else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}. Available: {list(VARIANTS)}")

    cfgs = {name: deep_merge(CONFIG, VARIANTS[name]) for name in names}

    root_dir = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    root_dir.mkdir(parents=True, exist_ok=True)
    h5_file = root_dir / "dataset.h5"

    print_plan(cfgs)

    if args.dry_run:
        print("\n[Dry run] 학습을 수행하지 않고 종료합니다.")
        return

    # 원본 데이터는 기준 config(2000점)로 1회만 생성해 모든 variant가 공유
    base_qs = build_qs(CONFIG["simulation"])
    if not h5_file.exists():
        print(f"📦 Source dataset missing. Generating at {h5_file} ...")
        simulate.generate_1layer_data(base_qs, CONFIG, h5_file)

    # step 예산 산정: 기준 variant의 (config epochs × steps/epoch)
    budget_steps = None
    if args.budget_mode == "steps" and args.epochs is None:
        base_cfg = deep_merge(CONFIG, VARIANTS[args.baseline])
        base_loader, _, _ = make_loaders(
            build_qs(base_cfg["simulation"]), base_cfg, h5_file,
            root_dir / "ablation" / args.baseline / "stats.pt",
        )
        budget_steps = base_cfg["training"]["epochs"] * steps_per_epoch(base_loader)
        print(f"\n[Budget] baseline={args.baseline}, total steps = {budget_steps:,} "
              f"(모든 variant를 이 step 수에 맞춤)")

    rows = []
    out_csv = root_dir / "ablation" / "ablation_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for name in names:
        row = run_variant(name, cfgs[name], root_dir, h5_file, budget_steps, args.epochs)
        rows.append(row)

        # 매 variant마다 중간 저장 (중단 대비)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[Save] Ablation summary → {out_csv}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    cols = ["variant", "q_points", "use_fourier", "augment", "params",
            "Thickness_MAE", "Roughness_MAE", "SLD_MAE", "Thickness_R2"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print("=" * 100)

    (root_dir / "ablation" / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main()
