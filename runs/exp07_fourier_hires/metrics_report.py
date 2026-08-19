"""
과제 보고용 확장 지표 산출기.

추론을 다시 돌리지 않는다 - ablation 평가가 남긴 evaluation_results_{tag}.csv에는
샘플별 True/Pred/Error가 그대로 있으므로 거기서 전부 계산한다.
추론 속도만 예외로, 실제 모델을 올려 측정한다(--benchmark).

산출물 (ablation/metrics/):
  metrics_extended_{tag}.csv   파라미터별 MAE/RMSE/MAPE/MedAE/P90/P95/bias/SD/R2
  metrics_tolerance_{tag}.csv  허용오차 내 샘플 비율
  metrics_by_thickness_{tag}.csv  두께 구간별 분해
  metrics_cost.csv             파라미터 수 / 학습 시간 / 수렴 epoch / 추론 처리량
  METRICS.md                   위 표를 마크다운으로 (과제에 붙여넣기용)

사용:
    python runs/exp07_fourier_hires/metrics_report.py
    python runs/exp07_fourier_hires/metrics_report.py --benchmark
    python runs/exp07_fourier_hires/metrics_report.py --only A_full E_exp02_like
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ablation import VARIANTS, build_qs, deep_merge, make_loaders
from config import CONFIG

# 보고서에서 previous/current로 부르는 쌍. 나머지는 ablation 요소로만 등장한다.
REPORT_LABELS = {"E_exp02_like": "previous (exp02-like)", "A_full": "current (exp07)"}

# (csv 접두사, 표시 이름, 단위)
PARAMS = [
    ("Thickness", "Thickness", "Å"),
    ("Roughness", "Roughness", "Å"),
    ("SLD", "SLD", "1e-6 Å⁻²"),
]

# 두께 구간. config의 param_ranges (10-1200 Å)를 물리적으로 의미 있는 구간으로 나눈다.
THICKNESS_BANDS = [(10, 50), (50, 100), (100, 300), (300, 600), (600, 1200)]

# 허용오차 정의. 상대(%)와 절대(단위) 중 파라미터 성격에 맞는 쪽을 쓴다.
TOLERANCES = {
    "Thickness": [("rel", 1.0), ("rel", 5.0), ("rel", 10.0), ("abs", 5.0), ("abs", 10.0)],
    "Roughness": [("abs", 0.5), ("abs", 1.0), ("abs", 2.0)],
    "SLD": [("rel", 1.0), ("rel", 5.0), ("rel", 10.0), ("abs", 1.0)],
}


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def core_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    abs_err = np.abs(err)
    rel = abs_err / (np.abs(true) + 1e-6) * 100
    return {
        "n": len(true),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MedAE": float(np.median(abs_err)),
        "P90_AE": float(np.percentile(abs_err, 90)),
        "P95_AE": float(np.percentile(abs_err, 95)),
        "Max_AE": float(abs_err.max()),
        "MAPE": float(rel.mean()),
        "MedAPE": float(np.median(rel)),
        "Bias": float(err.mean()),
        "SD": float(err.std()),
        "R2": r2(true, pred),
    }


def load_eval(var_dir: Path, tag: str) -> pd.DataFrame | None:
    path = var_dir / f"evaluation_results_{tag}.csv"
    if not path.exists():
        # 학습 직후 저장되는 무접미사 파일(clean 도메인)을 대체로 허용
        alt = var_dir / "evaluation_results.csv"
        if tag == "clean" and alt.exists():
            return pd.read_csv(alt)
        return None
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# 표 만들기
# -----------------------------------------------------------------------------
def build_extended(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, df in dfs.items():
        for key, name, unit in PARAMS:
            m = core_metrics(df[f"{key}_True"].to_numpy(), df[f"{key}_Pred"].to_numpy())
            rows.append({"variant": variant, "label": REPORT_LABELS.get(variant, ""),
                         "parameter": name, "unit": unit, **m})
    return pd.DataFrame(rows)


def build_tolerance(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, df in dfs.items():
        row: dict = {"variant": variant, "label": REPORT_LABELS.get(variant, "")}
        for key, name, unit in PARAMS:
            true = df[f"{key}_True"].to_numpy()
            abs_err = np.abs(df[f"{key}_Pred"].to_numpy() - true)
            for kind, thr in TOLERANCES[key]:
                if kind == "rel":
                    hit = abs_err <= np.abs(true) * thr / 100
                    col = f"{name}_within_{thr:g}pct"
                else:
                    hit = abs_err <= thr
                    col = f"{name}_within_{thr:g}{unit.split()[0] if unit != '1e-6 Å⁻²' else 'u'}"
                row[col] = float(hit.mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def build_by_thickness(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, df in dfs.items():
        t_true = df["Thickness_True"].to_numpy()
        for lo, hi in THICKNESS_BANDS:
            mask = (t_true >= lo) & (t_true < hi)
            if mask.sum() == 0:
                continue
            sub = df[mask]
            row = {"variant": variant, "label": REPORT_LABELS.get(variant, ""),
                   "band": f"{lo}-{hi} Å", "band_lo": lo, "band_hi": hi,
                   "n": int(mask.sum())}
            for key, name, _ in PARAMS:
                m = core_metrics(sub[f"{key}_True"].to_numpy(), sub[f"{key}_Pred"].to_numpy())
                row[f"{name}_MAE"] = m["MAE"]
                row[f"{name}_MAPE"] = m["MAPE"]
                row[f"{name}_R2"] = m["R2"]
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 비용 / 속도
# -----------------------------------------------------------------------------
def benchmark_inference(variant: str, ablation_dir: Path, h5_file: Path,
                        n_batches: int = 30) -> dict:
    """test 로더로 실제 추론 처리량을 잰다. 데이터 로딩을 빼고 순수 forward만 잰다."""
    import time

    import torch
    from xrr_model import XRRPhysicsModel

    var_dir = ablation_dir / variant
    ckpt_path = var_dir / "best.pt"
    if not ckpt_path.exists():
        return {}

    cfg = deep_merge(CONFIG, VARIANTS[variant])
    qs = build_qs(cfg["simulation"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt.get("config", {}).get("model_args", {})
    model = XRRPhysicsModel(**model_args).to(device).eval()

    (loader,) = make_loaders(qs, cfg, h5_file, var_dir / "stats.pt",
                             modes=["test"], num_workers=0)

    # 배치를 미리 GPU로 올려두고 forward만 반복 측정한다
    batches = []
    for i, (X, _) in enumerate(loader):
        batches.append(X.to(device))
        if i + 1 >= n_batches:
            break
    if not batches:
        return {}

    with torch.inference_mode():
        for _ in range(3):                       # warm-up (cudnn autotune 포함)
            model(batches[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_samples = 0
        for X in batches:
            model(X)
            n_samples += X.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return {
        "infer_samples_per_s": n_samples / elapsed,
        "infer_ms_per_sample": elapsed / n_samples * 1000,
        "infer_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "infer_batch_size": batches[0].shape[0],
    }


def build_cost(variants: list[str], ablation_dir: Path, h5_file: Path,
               benchmark: bool) -> pd.DataFrame:
    summary_path = ablation_dir / "ablation_summary.csv"
    summary = pd.read_csv(summary_path).set_index("variant") if summary_path.exists() else None

    rows = []
    for v in variants:
        var_dir = ablation_dir / v
        cfg = deep_merge(CONFIG, VARIANTS[v])
        row: dict = {
            "variant": v,
            "label": REPORT_LABELS.get(v, ""),
            "q_points": cfg["simulation"]["q_points"],
            "depth": cfg["model"]["depth"],
            "use_fourier": cfg["model"]["use_fourier"],
            "augment": cfg["training"]["augment"],
        }

        if summary is not None and v in summary.index:
            s = summary.loc[v]
            for col in ["params", "train_samples", "epochs_budget", "epochs_ran",
                        "best_epoch", "early_stopped", "steps_per_epoch",
                        "total_steps", "best_val_loss", "train_seconds"]:
                if col in summary.columns:
                    row[col] = s[col]

        # ablation_summary에 train_seconds가 없는 구버전 실행분은 로그 파일에서 보완
        time_log = var_dir / "train_time.json"
        if row.get("train_seconds") is None and time_log.exists():
            row["train_seconds"] = json.loads(time_log.read_text())["cumulative_seconds"]
        if row.get("train_seconds") is not None and not pd.isna(row["train_seconds"]):
            row["train_hours"] = float(row["train_seconds"]) / 3600

        if benchmark:
            print(f"  [Benchmark] {v} ...")
            row.update(benchmark_inference(v, ablation_dir, h5_file))

        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 마크다운 리포트
# -----------------------------------------------------------------------------
def fmt(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    """pandas.to_markdown은 tabulate 의존이라 직접 만든다."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
        else:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else str(x))

    cols = list(out.columns)
    widths = [max(len(c), *(len(v) for v in out[c])) if len(out) else len(c) for c in cols]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
            for row in out.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *body])


def write_markdown(path: Path, tables: dict, budget_note: str):
    lines = [
        "# exp07 ablation — 확장 지표",
        "",
        budget_note,
        "",
        "평가 도메인 두 가지:",
        "- **clean** — 시뮬레이션 원본. augmentation 없이 학습한 variant에 유리하게 편향됨",
        "- **augmented** — 측정 노이즈(강도 스케일·q shift·해상도 smearing)가 실린 조건. 실측에 가까움",
        "",
        "실측 성능을 대표하는 것은 augmented 쪽이다.",
        "",
        "### 지표 해석 주의",
        "",
        "- **Roughness의 MAPE/MedAPE는 쓰지 말 것.** 참값 범위가 0–15 Å이라 0 근처 샘플에서 "
        "상대오차가 발산한다. Roughness는 MAE·P90·허용오차 비율(절대값)로 판단한다.",
        "- **두께 구간별 R²는 구간 간 비교용이 아니다.** 구간을 자르면 참값 분산이 줄어 "
        "R²가 구조적으로 낮아진다(좁은 구간일수록 음수가 되기 쉬움). 구간별로는 MAE·MAPE를 본다.",
        "- Bias(평균오차)는 계통편향, SD는 산포다. MAE가 같아도 Bias가 크면 보정 가능한 오차다.",
        "",
    ]

    for tag in ["augmented", "clean"]:
        if f"extended_{tag}" not in tables:
            continue
        lines += [f"## 평가 도메인: {tag}", "",
                  "### 파라미터별 오차 지표", "", fmt(tables[f"extended_{tag}"]), "",
                  "### 허용오차 내 샘플 비율 (%)", "", fmt(tables[f"tolerance_{tag}"], "{:.2f}"), "",
                  "### 두께 구간별 분해", "", fmt(tables[f"by_thickness_{tag}"]), ""]

    if "cost" in tables:
        lines += ["## 비용 / 속도", "", fmt(tables["cost"]), ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Save] {path}")


def main():
    parser = argparse.ArgumentParser(description="exp07 확장 지표 산출")
    parser.add_argument("--only", nargs="+", default=None,
                        help=f"대상 variant (기본: 전체). 선택지: {list(VARIANTS)}")
    parser.add_argument("--domains", nargs="+", default=["augmented", "clean"],
                        choices=["augmented", "clean"])
    parser.add_argument("--benchmark", action="store_true",
                        help="추론 처리량을 실제 GPU에서 측정 (모델 로드 필요)")
    args = parser.parse_args()

    variants = args.only or list(VARIANTS)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}. Available: {list(VARIANTS)}")

    root = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    ablation_dir = root / "ablation"
    out_dir = ablation_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}

    for tag in args.domains:
        dfs = {}
        for v in variants:
            df = load_eval(ablation_dir / v, tag)
            if df is None:
                print(f"⚠️  {v}: evaluation_results_{tag}.csv 없음 — 건너뜀")
                continue
            dfs[v] = df
        if not dfs:
            continue

        print(f"\n[{tag}] {len(dfs)} variants, {len(next(iter(dfs.values()))):,} test samples")
        tables[f"extended_{tag}"] = build_extended(dfs)
        tables[f"tolerance_{tag}"] = build_tolerance(dfs)
        tables[f"by_thickness_{tag}"] = build_by_thickness(dfs)

        for kind in ["extended", "tolerance", "by_thickness"]:
            p = out_dir / f"metrics_{kind}_{tag}.csv"
            tables[f"{kind}_{tag}"].to_csv(p, index=False, encoding="utf-8-sig")
            print(f"  [Save] {p.name}")

    print("\n[cost]")
    tables["cost"] = build_cost(variants, ablation_dir, root / "dataset.h5", args.benchmark)
    tables["cost"].to_csv(out_dir / "metrics_cost.csv", index=False, encoding="utf-8-sig")
    print(f"  [Save] metrics_cost.csv")

    steps = tables["cost"]["total_steps"].max() if "total_steps" in tables["cost"] else None
    budget_note = (f"> 학습 예산: variant당 약 {int(steps):,} optimizer steps (step-matched)"
                   if steps and not pd.isna(steps) else "")
    write_markdown(out_dir / "METRICS.md", tables, budget_note)

    print(f"\n✅ 지표 산출 완료 → {out_dir}")


if __name__ == "__main__":
    main()
