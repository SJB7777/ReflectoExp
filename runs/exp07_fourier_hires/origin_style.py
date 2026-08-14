"""
OriginLab 스타일 matplotlib 설정.

Origin 기본 플롯의 시각적 규약을 따른다:
  - 사방이 닫힌 검은 프레임(box), 안쪽을 향한 major/minor tick
  - grid 없음 (Origin 기본값)
  - Arial 계열 sans-serif
  - 검은 테두리의 채워진 심볼
  - 테두리가 있는 불투명 범례

색은 계열(entity)에 고정 순서로 배정하고 절대 순환시키지 않는다. 아래 순서는
색각이상 분리도 검증을 통과한 것이므로(최악 인접쌍 protan ΔE 24.9) 순서를 바꾸지 말 것.
검은색은 계열 색이 아니라 축/텍스트/기준선 전용이다.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# 계열 색: 고정 순서. 5개를 넘기면 색을 만들어내지 말고 facet으로 쪼갤 것.
ORIGIN_COLORS: list[str] = [
    "#C00000",  # red
    "#0070C0",  # blue
    "#00A550",  # green
    "#7030A0",  # purple
    "#E08B00",  # orange
]

# 색과 짝지어 쓰는 2차 인코딩(색각이상/흑백 인쇄 대비). Origin 기본 심볼 순서.
ORIGIN_MARKERS: list[str] = ["s", "o", "^", "D", "v"]

# 축, 텍스트, 기준선 전용
INK = "#000000"
REFERENCE = "#000000"


def apply_origin_style() -> None:
    """전역 rcParams에 Origin 스타일 적용."""
    mpl.rcParams.update({
        # --- Font ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "mathtext.fontset": "dejavusans",

        # --- Frame: 사방이 닫힌 박스 ---
        "axes.linewidth": 1.2,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": True,
        "axes.spines.right": True,

        # --- Ticks: 안쪽, 사방 ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # --- Grid: Origin 기본은 없음 ---
        "axes.grid": False,

        # --- Marks ---
        "lines.linewidth": 1.5,
        "lines.markersize": 6.0,
        "lines.markeredgewidth": 1.0,
        "axes.prop_cycle": mpl.cycler(color=ORIGIN_COLORS),

        # --- Legend: 테두리 있는 불투명 박스 ---
        "legend.frameon": True,
        "legend.edgecolor": INK,
        "legend.facecolor": "white",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.borderpad": 0.4,

        # --- Surface ---
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 100,
    })


def style_axes(ax: plt.Axes, minor: bool = True) -> plt.Axes:
    """개별 Axes에 프레임/tick 규약 적용. rcParams를 우회하는 플롯(seaborn 등)용."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color(INK)

    ax.tick_params(which="major", direction="in", length=5.0, width=1.2,
                   top=True, right=True, color=INK)
    ax.tick_params(which="minor", direction="in", length=2.5, width=1.0,
                   top=True, right=True, color=INK)
    if minor:
        ax.minorticks_on()
    ax.grid(False)
    return ax


def series_style(index: int) -> dict:
    """계열 index에 대한 색 + 심볼. 색만으로 구분되지 않도록 항상 함께 쓴다."""
    return {
        "color": ORIGIN_COLORS[index % len(ORIGIN_COLORS)],
        "marker": ORIGIN_MARKERS[index % len(ORIGIN_MARKERS)],
    }
