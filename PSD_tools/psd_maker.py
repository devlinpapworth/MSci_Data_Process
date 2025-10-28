

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import Iterable, Sequence, Optional

def d_at_percent(
    sizes_um: Sequence[float],
    percent_passing: Sequence[float],
    p: float
) -> float:
    """
    Return Dx (e.g., D10, D50, D90) for a PSD curve via linear interpolation in % passing.
    sizes_um must be strictly increasing; percent_passing must be non-decreasing.
    """
    x = np.asarray(sizes_um, dtype=float)
    y = np.asarray(percent_passing, dtype=float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("sizes_um must be strictly increasing.")
    if np.any(np.diff(y) < 0):
        raise ValueError("percent_passing must be non-decreasing (cumulative).")
    if not (0 <= p <= 100):
        raise ValueError("p must be in [0, 100].")

    return float(np.interp(p, y, x))

def plot_psd(
    sizes_um: Sequence[float],
    percent_passing: Sequence[float],
    *,
    title: str = "Particle Size Distribution (PSD)",
    x_major_ticks: Iterable[float] = (0.1, 1, 10, 100),
    x_limits: Optional[tuple[float, float]] = (0.1, 100),
    show_grid: bool = True,
    annotate_dx: Iterable[int] | None = (10, 50, 90),
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict[str, float]:
    """
    Plot a PSD curve with a logarithmic x-axis.
    Returns a dict of any annotated Dx values (e.g., {'D10': ..., 'D50': ...}).

    Parameters
    ----------
    sizes_um : ascending particle sizes (µm)
    percent_passing : cumulative % passing, non-decreasing
    x_major_ticks : where to place the log ticks (default: 0.1, 1, 10, 100)
    x_limits : x-axis limits (log scale)
    annotate_dx : which Dx to annotate (None to skip)
    save_path : path to save the figure (e.g., 'psd.png'), None to skip
    show : show the figure (True/False)
    """
    x = np.asarray(sizes_um, dtype=float)
    y = np.asarray(percent_passing, dtype=float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("sizes_um must be strictly increasing.")
    if np.any(np.diff(y) < 0):
        raise ValueError("percent_passing must be non-decreasing (cumulative).")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(x, y, "-o", linewidth=2, markersize=4)

    if x_limits:
        ax.set_xlim(*x_limits)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Particle size (µm)")
    ax.set_ylabel("Percent passing (%)")
    ax.set_title(title)

    # Ticks exactly at 0.1, 1, 10, 100 (or whatever is passed)
    xticks = list(x_major_ticks)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    if show_grid:
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.2)

    dx_out: dict[str, float] = {}
    if annotate_dx:
        for p in annotate_dx:
            d = d_at_percent(x, y, p)
            dx_out[f"D{p}"] = d
            ax.axhline(p, color="grey", linestyle="--", linewidth=1)
            ax.axvline(d, color="grey", linestyle="--", linewidth=1)
            ax.text(d, p + 2, f"D{p} = {d:.3g} µm", ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dx_out
