"""
Ablation 결과 비교 그림 (OriginLab 스타일).

clean / augmented 두 평가 도메인의 요약 CSV를 읽어 리포트용 그림을 만든다.
두 도메인을 함께 그리는 것이 핵심이다 - clean 평가만 보면 augmentation 없이 학습한
variant가 유리해 보이는데(학습/평가 분포 일치), 이는 일반화 성능이 아니다.

사용:
    python runs/exp07_fourier_hires/plot_ablation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CONFIG
from origin_style import (
    INK,
    ORIGIN_COLORS,
    ORIGIN_MARKERS,
    apply_origin_style,
    style_axes,
)

apply_origin_style()

PARAMS = [
    ("Thickness", "Thickness (Å)", "MAE (Å)"),
    ("Roughness", "Roughness (Å)", "MAE (Å)"),
    ("SLD", r"SLD (10$^{-6}$ Å$^{-2}$)", r"MAE (10$^{-6}$ Å$^{-2}$)"),
]

# 도메인은 2개뿐이므로 계열 색 2개만 사용 (고정 순서).
# 심볼도 함께 바꿔 색만으로 구분되지 않게 한다.
DOMAIN_STYLE = {
    "clean": {"color": ORIGIN_COLORS[1], "marker": "o", "label": "Clean simulation"},
    "augmented": {"color": ORIGIN_COLORS[0], "marker": "s", "label": "With measurement noise"},
}

CONNECTOR = "#9A9A9A"


def _dumbbell(ax, y, lo, hi):
    """두 도메인 값을 잇는 연결선. 값이 아니라 이동량을 읽게 하는 보조선."""
    ax.plot([lo, hi], [y, y], color=CONNECTOR, lw=1.2, zorder=1)


def load_summaries(ablation_dir: Path) -> dict[str, pd.DataFrame]:
    """도메인별 요약 CSV 로드. 없으면 조용히 건너뛴다."""
    candidates = {
        "clean": ["ablation_summary_clean.csv", "ablation_summary.csv"],
        "augmented": ["ablation_summary_augmented.csv"],
    }
    out = {}
    for domain, names in candidates.items():
        for fname in names:
            path = ablation_dir / fname
            if path.exists():
                out[domain] = pd.read_csv(path)
                print(f"[Load] {domain:<9} <- {fname}")
                break
        else:
            print(f"[Skip] {domain}: summary CSV not found")
    return out


def plot_mae_by_variant(dfs: dict[str, pd.DataFrame], save_path: Path):
    """
    파라미터별 MAE를 variant × 평가도메인 dumbbell 점 그래프로.

    막대를 쓰지 않는 이유: 값의 범위가 두 자릿수를 넘어 로그 축이 필요한데,
    로그 축 위의 막대는 길이가 크기에 비례하지 않아 잘못 읽힌다.
    """
    if not dfs:
        return

    order = list(next(iter(dfs.values()))["variant"])
    domains = [d for d in ("clean", "augmented") if d in dfs]
    y = np.arange(len(order))[::-1]  # 첫 variant가 위로 오도록

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4.9 * len(PARAMS), 4.2),
                             sharey=True)

    for ax, (key, title, xlabel) in zip(axes, PARAMS, strict=True):
        style_axes(ax)
        col = f"{key}_MAE"
        series = {d: dfs[d].set_index("variant") for d in domains}

        if len(domains) == 2:
            for yi, v in zip(y, order, strict=True):
                lo = series["clean"].loc[v, col]
                hi = series["augmented"].loc[v, col]
                _dumbbell(ax, yi, lo, hi)

        for domain in domains:
            df = series[domain]
            vals = [df.loc[v, col] if v in df.index else np.nan for v in order]
            ax.plot(vals, y, ls="none",
                    marker=DOMAIN_STYLE[domain]["marker"], ms=8,
                    mfc=DOMAIN_STYLE[domain]["color"], mec=INK, mew=0.8,
                    label=DOMAIN_STYLE[domain]["label"], zorder=3)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="y", which="minor", left=False, right=False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)

    # 패널 안에 두면 데이터 점을 가리므로 그림 상단에 가로로 배치
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Ablation: mean absolute error by variant and evaluation domain",
                 fontsize=13, y=1.06)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Save] {save_path.name}")


def plot_domain_gap(dfs: dict[str, pd.DataFrame], save_path: Path):
    """
    clean → augmented 로 도메인이 바뀔 때 오차가 몇 배 늘어나는지.
    augmentation의 효과가 가장 직접적으로 드러나는 지표다.
    """
    if not {"clean", "augmented"} <= dfs.keys():
        print("[Skip] domain gap plot: needs both clean and augmented summaries")
        return

    clean = dfs["clean"].set_index("variant")
    aug = dfs["augmented"].set_index("variant")
    order = [v for v in clean.index if v in aug.index]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    style_axes(ax)

    y = np.arange(len(order))[::-1]
    offset = 0.18

    for i, (key, title, _) in enumerate(PARAMS):
        col = f"{key}_MAE"
        ratio = [aug.loc[v, col] / clean.loc[v, col] for v in order]
        ax.plot(ratio, y + (i - (len(PARAMS) - 1) / 2) * offset, ls="none",
                marker=ORIGIN_MARKERS[i], ms=8,
                mfc=ORIGIN_COLORS[i], mec=INK, mew=0.8, label=title, zorder=3)

    ax.axvline(1.0, color=INK, linestyle="--", lw=1.2)
    ax.text(1.05, len(order) - 0.6, "no degradation", va="top", ha="left",
            fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("MAE ratio (with noise / clean)")
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_title("Robustness to measurement noise (lower is better)")
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Save] {save_path.name}")


def plot_r2_panel(dfs: dict[str, pd.DataFrame], save_path: Path):
    """R²는 1에 붙어 있으므로 1−R²를 로그 축으로 그려야 차이가 보인다."""
    if not dfs:
        return

    order = list(next(iter(dfs.values()))["variant"])
    domains = [d for d in ("clean", "augmented") if d in dfs]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    style_axes(ax)

    y = np.arange(len(order))[::-1]
    series = {d: dfs[d].set_index("variant") for d in domains}

    if len(domains) == 2:
        for yi, v in zip(y, order, strict=True):
            _dumbbell(ax, yi,
                      1.0 - series["clean"].loc[v, "Thickness_R2"],
                      1.0 - series["augmented"].loc[v, "Thickness_R2"])

    for domain in domains:
        df = series[domain]
        vals = [1.0 - df.loc[v, "Thickness_R2"] if v in df.index else np.nan
                for v in order]
        ax.plot(vals, y, ls="none",
                marker=DOMAIN_STYLE[domain]["marker"], ms=8,
                mfc=DOMAIN_STYLE[domain]["color"], mec=INK, mew=0.8,
                label=DOMAIN_STYLE[domain]["label"], zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(r"$1 - R^2$  (thickness)")
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_title("Unexplained variance in thickness (lower is better)")
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    # 상단 두 행은 x가 작아 우상단이 비어 있다 (하단은 E의 노이즈 값이 차지)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Save] {save_path.name}")


def main():
    root = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    ablation_dir = root / "ablation"
    if not ablation_dir.exists():
        raise SystemExit(f"Ablation directory not found: {ablation_dir}")

    dfs = load_summaries(ablation_dir)
    if not dfs:
        raise SystemExit("No summary CSV found - run ablation.py first")

    fig_dir = ablation_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_mae_by_variant(dfs, fig_dir / "ablation_mae.png")
    plot_r2_panel(dfs, fig_dir / "ablation_r2.png")
    plot_domain_gap(dfs, fig_dir / "ablation_domain_gap.png")

    print(f"\n✅ Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
