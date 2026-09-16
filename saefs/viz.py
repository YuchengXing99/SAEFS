"""Plotting helpers and the figure palette used by the reproduction notebook.

Static matplotlib figures, one light surface, a fixed categorical hue order that
is assigned by slot (never cycled) and validated for colour-vision deficiency.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "SERIES", "SEQ_BLUE", "DIVERGING", "SURFACE", "INK", "INK_2", "MUTED", "GRID", "BASELINE",
    "set_style", "style_axes", "direct_label", "km_curves", "risk_groups",
]

# Categorical slots, assigned in this order -- see references/palette.md
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Single-hue sequential ramp (light -> dark) for magnitude
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
# Diverging: warm/cool poles with a neutral midpoint
DIVERGING = {"low": "#2a78d6", "mid": "#f0efec", "high": "#e34948"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def set_style() -> None:
    """Recessive chrome, thin marks, no chart junk."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 120,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "text.color": INK,
    })


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "",
               subtitle: str = "", grid_axis: Optional[str] = "y") -> None:
    """Apply the title / subtitle / grid conventions to one axes.

    The subtitle sits in the gap the title pad opens up, so the two never
    collide however many lines the title runs to.
    ``grid_axis=None`` removes the grid entirely (heatmaps, image panels).
    """
    if title:
        n_lines = title.count("\n") + 1
        ax.set_title(title, color=INK, pad=14 + 13 * n_lines if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", fontsize=9, color=MUTED, va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=False)
    if grid_axis in ("x", "y"):
        ax.grid(axis=grid_axis, visible=True)
    ax.tick_params(length=0)


def direct_label(ax, x, y, text: str, color: str, **kwargs) -> None:
    """Label a series at its end instead of relying on the legend alone."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=9, va="center", fontweight="semibold", **kwargs)


def risk_groups(risk: np.ndarray, n_groups: int = 2) -> np.ndarray:
    """Split cases into equal-sized risk groups (0 = lowest predicted risk)."""
    ranks = np.argsort(np.argsort(risk))
    return (ranks * n_groups // len(risk)).astype(int)


def km_curves(
    ax,
    durations: np.ndarray,
    events: np.ndarray,
    groups: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    show_ci: bool = True,
    censor_ticks: bool = True,
):
    """Kaplan-Meier curves, optionally stratified, with a multivariate log-rank test.

    Args:
        durations: observed time (days).
        events: 1 if the event was observed, 0 if censored.
        groups: optional integer group index per case.

    Returns:
        The log-rank p-value, or ``None`` when a single curve is drawn.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(int)
    if groups is None:
        groups = np.zeros(len(durations), dtype=int)
    groups = np.asarray(groups)

    keys = np.unique(groups)
    colors = colors or SERIES
    labels = labels or [f"group {k}" for k in keys]

    ends = []
    for i, key in enumerate(keys):
        m = groups == key
        kmf = KaplanMeierFitter().fit(durations[m], events[m], label=labels[i])
        color = colors[i % len(colors)]
        kmf.plot_survival_function(
            ax=ax, color=color, ci_show=show_ci, ci_alpha=0.10,
            show_censors=censor_ticks,
            censor_styles={"ms": 4, "marker": "|", "alpha": 0.6},
            linewidth=2.0,
        )
        ends.append((float(durations[m].max()), float(kmf.survival_function_.iloc[-1, 0]),
                     f"{labels[i]} (n={m.sum()})", color))

    ax.set_ylim(0, 1.02)
    ax.legend().set_visible(False)

    # survival curves finish close together; spread the end labels so they stay legible
    ends.sort(key=lambda e: e[1])
    min_gap = 0.055 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(1, len(ends)):
        x, y, text, color = ends[i]
        if y - ends[i - 1][1] < min_gap:
            ends[i] = (x, ends[i - 1][1] + min_gap, text, color)
    for x, y, text, color in ends:
        direct_label(ax, x, y, text, color)

    if len(keys) > 1:
        return float(multivariate_logrank_test(durations, groups, events).p_value)
    return None
