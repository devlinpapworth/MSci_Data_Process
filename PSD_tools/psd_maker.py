# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------
# Public convenience: consistent default colors for sample codes/names
# Extend/override in your main() as needed.
# ---------------------------------------------------------------------
DEFAULT_SAMPLE_COLORS: Dict[str, str] = {
    "Si_F": "#1f77b4",       # blue
    "Si_M": "#ff7f0e",       # orange
    "Si_C": "#2ca02c",       # green
    "Si_Rep": "#d62728",     # red
    "Si_Rep_new": "#9467bd", # purple
    "RT_As Received": "#000000",  # manual overlay (black by default)
}

# ---------------------------------------------------------------------
# Single-curve helpers
# ---------------------------------------------------------------------
def d_at_percent(
    sizes_um: Sequence[float],
    percent_passing: Sequence[float],
    p: float
) -> float:
    """
    Return the particle size (Um) at cumulative percent passing p.
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
    x_major_ticks: Iterable[float] = (0.1, 1, 10, 100, 1000),
    x_limits: Optional[Tuple[float, float]] = (0.1, 1000),
    show_grid: bool = True,
    annotate_dx: Iterable[int] | None = None,  # keep None to avoid drawing labels
    save_path: Optional[str] = None,
    show: bool = True,
    label: Optional[str] = None,               # NEW: used for legend + color lookup
    color_map: Optional[Dict[str, str]] = None # NEW: consistent color control
) -> Dict[str, float]:
    """
    ovided).
    """
    x = np.asarray(sizes_um, dtype=float)
    y = np.asarray(percent_passing, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("sizes_um must be strictly increasing.")
    if np.any(np.diff(y) < 0):
        raise ValueError("percent_passing must be non-decreasing.")

    # choose color
    if color_map is not None and label is not None:
        curve_color = color_map.get(label, "#808080")
    else:
        curve_color = "#1f77b4"  # default matplotlib blue

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(x, y, "-o", linewidth=2, markersize=4, color=curve_color, label=label)

    if x_limits:
        ax.set_xlim(*x_limits)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Particle size (\u03bcm)")
    ax.set_ylabel("Percent passing (%)")
    ax.set_title(title)
    ax.set_xticks(list(x_major_ticks))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    if show_grid:
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.2)

    # compute (but do not draw) D-values
    dx_out: Dict[str, float] = {}
    if annotate_dx:
        for p in annotate_dx:
            d = d_at_percent(x, y, p)
            dx_out[f"D{p}"] = d

    if label is not None:
        ax.legend(title="Sample Code", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return dx_out

# ---------------------------------------------------------------------
# All-samples (PSD_Full)
# ---------------------------------------------------------------------
def _extract_numeric_size(colname: str) -> Optional[float]:
    """
    .
    """
    for token in str(colname).split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def plot_all_psd(
    excel_path: str,
    sheet_name: str = "PSD_Full",
    *,
    title: str = "Particle Size Distribution",
    x_limits: Tuple[float, float] = (0.1, 1000),
    x_major_ticks: Iterable[float] = (0.1, 1, 10, 100, 1000),
    show_grid: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    legend: bool = True,
    include_samples: Optional[Iterable[Any]] = None,   # filter which rows to draw
    color_map: Optional[Dict[str, str]] = None,        # consistent per-sample colors
    id_col_override: Optional[str] = None              # force ID column if needed
) -> None:
    def to_cumulative(y_raw: np.ndarray) -> np.ndarray:
        y = np.nan_to_num(y_raw.astype(float), nan=0.0)
        if np.all(np.diff(y) >= -1e-6) and np.nanmax(y) <= 100 + 1e-6:
            return np.clip(y, 0, 100)
        tot = y.sum()
        return np.cumsum(y * (100.0 / tot)) if tot > 0 else y

    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")

    # --- Decide which ID column to use ---
    if id_col_override and id_col_override in df.columns:
        id_col = id_col_override
    else:
        id_col = "Sample Code" if "Sample Code" in df.columns else (
            "Sample Name" if "Sample Name" in df.columns else None
        )
    if id_col is None:
        raise ValueError("Neither 'Sample Code' nor 'Sample Name' found in the sheet.")

    # Normalize the ID column for robust matching
    df[id_col] = df[id_col].astype(str).str.strip()

    # --- Extract PSD size columns ---
    psd_cols: List[str] = []
    sizes_um: List[float] = []
    for c in df.columns:
        v = _extract_numeric_size(c)
        if v is not None:
            psd_cols.append(c)
            sizes_um.append(v)
    if not psd_cols:
        raise ValueError("No numeric PSD size columns found in the sheet.")

    order = np.argsort(sizes_um)
    sizes_um = np.asarray(sizes_um, dtype=float)[order]
    psd_cols = [psd_cols[i] for i in order]

    # --- Build normalized include set and filter rows ---
    allow_set = None
    if include_samples is not None:
        allow_set = {str(s).strip().casefold() for s in include_samples}
        mask = df[id_col].str.strip().str.casefold().isin(allow_set)
        df = df[mask]

    # setup figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    cidx = 0

    def resolve_color(label: str) -> Any:
        nonlocal cidx
        if color_map is not None:
            return color_map.get(label, "#808080")
        color = cmap(cidx % 20)
        cidx += 1
        return color

    # plot each PSD curve
    for _, row in df.iterrows():
        label = str(row[id_col]).strip()
        y_vals = row[psd_cols].to_numpy(dtype=float)
        if np.isnan(y_vals).all():
            continue
        y_cum = to_cumulative(y_vals)
        ax.semilogx(sizes_um, y_cum, linewidth=2, label=label, color=resolve_color(label))

    # --- Manual overlay ONLY if explicitly requested via include_samples ---
    manual_label = "RT_As Received"
    should_plot_manual = (
        include_samples is not None
        and (manual_label.strip().casefold() in allow_set)
    )
    if should_plot_manual:
        manual_sizes = np.array(
            [0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 26, 38, 53, 75, 106, 150, 212, 300, 425, 600, 1000],
            dtype=float,
        )
        manual_percent = np.array(
            [2.826, 4.277, 7.206, 12.586, 17.421, 25.123, 31.079, 40.088, 46.615,
             55.289, 62.971, 69.272, 75.268, 80.019, 84.453, 88.355, 91.703,
             94.637, 97.28, 99.165, 99.914, 100],
            dtype=float,
        )
        mo = np.argsort(manual_sizes)
        manual_sizes = manual_sizes[mo]
        manual_cum = to_cumulative(manual_percent[mo])

        manual_color = (color_map.get(manual_label, "black") if color_map is not None else "black")
        ax.semilogx(
            manual_sizes, manual_cum,
            linewidth=3,
            color=manual_color,
            label=manual_label,
        )

    # formatting
    ax.set_xlim(*x_limits)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Particle size (\u03bcm)")
    ax.set_ylabel("Cumulative percent undersize (%)")
    ax.set_title(title)
    ax.set_xticks(list(x_major_ticks))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    if show_grid:
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.2)
    if legend:
        ax.legend(title=id_col, fontsize=9, ncol=1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
