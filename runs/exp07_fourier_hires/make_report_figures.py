"""
발표/보고서용 그림 뽑기 (OriginLab 스타일).

각 모델의 성능 그림을 독립적으로 만든다. variant 간 비교 그림은 plot_ablation.py 쪽.

추론을 다시 돌리지 않는다 - ablation 평가가 남긴 evaluation_results_*.csv에
True/Pred/Error가 이미 들어 있으므로 그림만 다시 그린다. 학습곡선만 best.pt에서 읽는다.

사용:
    python runs/exp07_fourier_hires/make_report_figures.py
    python runs/exp07_fourier_hires/make_report_figures.py --only exp07 exp02
"""

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import CONFIG
from evaluate import (
    PARAM_NAMES,
    calculate_r2,
    plot_error_heatmap,
    save_correlation_plot,
    save_history_plot,
)
from origin_style import INK, ORIGIN_COLORS, apply_origin_style, style_axes

apply_origin_style()

# 출력 이름 -> ablation variant 폴더
# exp02는 원본 데이터가 소실되어 재현 불가하므로 동등 조건 variant로 대체한다
# (q 200점, Fourier 없음, augmentation 없음).
LABELS: dict[str, str] = {
    "current": "A_full",
    "previous": "E_exp02_like",
    "current_nofourier": "B_no_fourier",
    "current_noaug": "C_no_augment",
    "current_lowres": "D_lowres_200",
}

# 그림 안에 찍히는 이름. 발표는 이전 모델 / 현재 모델 대비로 구성하므로
# 실험 번호 대신 Previous / Current로 표기한다.
DISPLAY: dict[str, str] = {
    "current": "Current",
    "previous": "Previous",
    "current_nofourier": "Current (no Fourier)",
    "current_noaug": "Current (no augmentation)",
    "current_lowres": "Current (200 q-points)",
}

DOMAIN_TITLE = {"clean": "clean simulation", "augmented": "with measurement noise"}


def metrics_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """평가 CSV에서 파라미터별 지표 재계산."""
    rows = []
    for name in PARAM_NAMES:
        clean = name.split(" (")[0]
        true = df[f"{clean}_True"].to_numpy()
        pred = df[f"{clean}_Pred"].to_numpy()
        err = df[f"{clean}_Error"].to_numpy()
        rows.append({
            "parameter": clean,
            "MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "MAPE_%": np.mean(np.abs(err) / (np.abs(true) + 1e-6)) * 100,
            "R2": calculate_r2(true, pred),
        })
    return pd.DataFrame(rows)


def plot_error_vs_thickness(df: pd.DataFrame, save_path: Path, title: str):
    """
    두께에 따른 절대 오차. 어느 두께 구간에서 모델이 무너지는지 보여준다.
    점이 수천 개라 그대로 뿌리면 뭉치므로, 구간 중앙값과 사분위 범위를 함께 그린다.
    """
    thick = df["Thickness_True"].to_numpy()
    abs_err = np.abs(df["Thickness_Error"].to_numpy())

    n_bins = 20
    edges = np.linspace(thick.min(), thick.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(thick, edges) - 1, 0, n_bins - 1)

    med, q1, q3 = [], [], []
    for b in range(n_bins):
        vals = abs_err[idx == b]
        if len(vals) == 0:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
        else:
            med.append(np.median(vals))
            q1.append(np.percentile(vals, 25))
            q3.append(np.percentile(vals, 75))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style_axes(ax)

    ax.scatter(thick, abs_err, s=6, color=ORIGIN_COLORS[1], alpha=0.15,
               edgecolor="none", zorder=1)
    ax.fill_between(centers, q1, q3, color=ORIGIN_COLORS[0], alpha=0.20,
                    linewidth=0, zorder=2, label="Interquartile range")
    ax.plot(centers, med, color=ORIGIN_COLORS[0], marker="o", ms=5,
            mec=INK, mew=0.6, zorder=3, label="Median")

    ax.set_yscale("log")
    # 0에 가까운 소수의 오차가 축을 6 decade로 늘려 중앙값/IQR을 짓눌러 버린다.
    # 하한을 1 퍼센타일로 잘라내되, 잘린 비율을 그림 안에 밝힌다.
    floor = np.percentile(abs_err, 1)
    below = float(np.mean(abs_err < floor) * 100)
    ax.set_ylim(bottom=floor, top=abs_err.max() * 1.5)
    ax.text(0.99, 0.02, f"{below:.0f}% of points below axis",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)

    ax.set_xlabel("True thickness (Å)")
    ax.set_ylabel("Absolute thickness error (Å)")
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Save] {save_path.name}")


def build_for(label: str, variant: str, ablation_dir: Path, out_dir: Path,
              domains: list[str], clip_axes: bool):
    var_dir = ablation_dir / variant
    if not var_dir.exists():
        print(f"[Skip] {label}: {var_dir} not found")
        return

    shown = DISPLAY.get(label, label)
    print(f"\n=== {label}  (variant: {variant}, shown as '{shown}') ===")
    summary = []

    for domain in domains:
        csv_path = var_dir / f"evaluation_results_{domain}.csv"
        if not csv_path.exists():
            print(f"  [Skip] {domain}: {csv_path.name} not found")
            continue

        df = pd.read_csv(csv_path)

        head = f"{shown} — {DOMAIN_TITLE[domain]}"
        save_correlation_plot(df, PARAM_NAMES,
                              out_dir / f"{label}_parity_{domain}.png",
                              title_prefix=head, clip_axes=clip_axes)
        plot_error_heatmap(df, save_path=out_dir / f"{label}_heatmap_{domain}.png",
                           title_prefix=head)
        plot_error_vs_thickness(
            df, out_dir / f"{label}_error_vs_thickness_{domain}.png",
            f"{head}: thickness error vs film thickness",
        )

        m = metrics_from_df(df)
        m.insert(0, "domain", domain)
        summary.append(m)

    # 학습곡선은 체크포인트의 history에서
    ckpt_path = var_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        history = ckpt.get("history", {})
        if history:
            save_history_plot(history, out_dir / f"{label}_learning_curve.png",
                              title_prefix=shown)

    # 곡선 재구성 그림은 GenX 시뮬레이션이 필요하므로 기존 산출물을 가져온다
    recon = var_dir / "reconstruction_analysis.png"
    if recon.exists():
        shutil.copy2(recon, out_dir / f"{label}_reconstruction.png")
        print(f"  [Copy] {label}_reconstruction.png")

    if summary:
        table = pd.concat(summary, ignore_index=True)
        table.to_csv(out_dir / f"{label}_metrics.csv", index=False, encoding="utf-8-sig")
        print(f"  [Save] {label}_metrics.csv")
        print(table.to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"))


def main():
    parser = argparse.ArgumentParser(description="Report figure builder")
    parser.add_argument("--only", nargs="+", default=["current", "previous"],
                        help=f"출력할 라벨 (기본: current previous). 선택지: {list(LABELS)}")
    parser.add_argument("--all", action="store_true", help="정의된 라벨 전체 출력")
    parser.add_argument("--clean", action="store_true",
                        help="출력 폴더를 비우고 새로 생성 (이름 규칙이 바뀐 뒤 잔여 파일 제거용)")
    parser.add_argument("--domains", nargs="+", default=["augmented"],
                        choices=["clean", "augmented"],
                        help="평가 도메인 (기본: augmented). 절대 성능이 아니라 "
                             "측정 조건에서의 성능을 보여주는 쪽이 augmented")
    parser.add_argument("--no-clip", action="store_true",
                        help="축 자르기 없이 전체 범위로 그리기")
    args = parser.parse_args()

    labels = list(LABELS) if args.all else args.only
    unknown = [x for x in labels if x not in LABELS]
    if unknown:
        raise SystemExit(f"Unknown label(s): {unknown}. Available: {list(LABELS)}")

    root = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    ablation_dir = root / "ablation"
    out_dir = ablation_dir / "report_figures"
    if args.clean and out_dir.exists():
        removed = len(list(out_dir.iterdir()))
        shutil.rmtree(out_dir)
        print(f"[Clean] removed {removed} file(s) from {out_dir.name}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        build_for(label, LABELS[label], ablation_dir, out_dir,
                  domains=args.domains, clip_axes=not args.no_clip)

    write_readme(out_dir, labels, args.domains)
    print(f"\n✅ Figures written to {out_dir}")


def write_readme(out_dir: Path, labels: list[str], domains: list[str]):
    """그림이 무엇을 재는지, 무엇을 재지 않는지 함께 둔다."""
    lines = [
        "# Report figures",
        "",
        "OriginLab 스타일. `make_report_figures.py`로 생성.",
        "",
        "## 파일 이름",
        "",
        "| 접두사 | 그림에 표기되는 이름 | ablation variant |",
        "|---|---|---|",
    ]
    for label in labels:
        lines.append(f"| `{label}` | {DISPLAY.get(label, label)} | `{LABELS[label]}` |")

    lines += [
        "",
        f"각 모델마다: `_parity_`, `_heatmap_`, `_error_vs_thickness_` ({'/'.join(domains)}),",
        "`_learning_curve`, `_reconstruction`, `_metrics.csv`.",
        "",
        "## 평가 도메인",
        "",
        f"이번 출력: **{', '.join(domains)}**",
        "",
        "- **clean** — 노이즈 없는 시뮬레이션",
        "- **augmented** — footprint, 기기 분해능, I0 변동, 배경, Poisson 노이즈, q축 정렬오차 적용",
        "",
        "clean 지표만 제시하면 augmentation 없이 학습한 모델이 유리하게 보인다.",
        "학습 분포와 평가 분포가 같기 때문이며 일반화 성능이 아니다.",
        "clean 수치를 함께 쓰려면 `--domains clean augmented`로 재생성할 것.",
        "",
        "## 축 범위",
        "",
        "parity는 참값 범위로, 오차 히스토그램은 0.5–99.5 퍼센타일로 자른다.",
        "소수의 큰 이상치가 축을 늘려 본체를 뭉개기 때문이다.",
        "**MAE·RMSE·R² 등 지표는 항상 전체 데이터로 계산하며, 잘리는 것은 축뿐이다.**",
        "잘려 나간 점의 비율은 각 그림 안에 표기된다. 전체 범위가 필요하면 `--no-clip`.",
        "",
        "## 슬라이드 각주 (그대로 붙여 쓸 것)",
        "",
        "> Previous는 exp02 원본 실험이 아니라 조건 근사다. exp02의 데이터셋과 체크포인트가",
        "> 소실되어, exp07 코드베이스에서 q 200점 / Fourier feature 없음 / augmentation 없음 /",
        "> depth 4 조건으로 재현했다. 원본 exp02와는 시뮬레이터(refnx vs GenX), 층 구조",
        "> (SiO2층 유무), 두께 범위(5–200 Å vs 10–1200 Å)가 다르다.",
        "",
        "> 학습 예산은 기준 조건의 2 %(93,750 / 4,687,500 step)이며 모든 모델에 동일하게",
        "> 맞췄다. 모델 간 상대 비교용이고 절대 성능이 아니다.",
        "",
        "> 두 평가 도메인 모두 시뮬레이션이다. 실측 데이터 검증은 수행하지 않았다.",
        "",
        "## 주의",
        "",
        "거칠기 MAPE는 참값이 0에 가까운 샘플 때문에 분모가 붕괴해 40–350 %로 나온다.",
        "거칠기는 MAE와 R²로 보고할 것.",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Save] {path.name}")


if __name__ == "__main__":
    main()
