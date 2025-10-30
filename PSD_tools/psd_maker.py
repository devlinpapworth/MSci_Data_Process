# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import Optional, Iterable, Sequence, Tuple, List

# ---- Single-curve helpers ----
def d_at_percent(sizes_um: Sequence[float], percent_passing: Sequence[float], p: float) -> float:
    x = np.asarray(sizes_um, dtype=float)
    y = np.asarray(percent_passing, dtype=float)
    if np.any(np.diff(x) <= 0): raise ValueError("sizes_um must be strictly increasing.")
    if np.any(np.diff(y) < 0):  raise ValueError("percent_passing must be non-decreasing (cumulative).")
    if not (0 <= p <= 100):     raise ValueError("p must be in [0, 100].")
    return float(np.interp(p, y, x))

def plot_psd(
    sizes_um: Sequence[float],
    percent_passing: Sequence[float],
    *,
    title: str = "Particle Size Distribution (PSD)",
    x_major_ticks: Iterable[float] = (0.1, 1, 10, 100, 1000),
    x_limits: Optional[Tuple[float, float]] = (0.1, 1000),
    show_grid: bool = True,
    annotate_dx: Iterable[int] | None = None,   # set to None to avoid D10/D50/D90
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict[str, float]:
    x = np.asarray(sizes_um, dtype=float)
    y = np.asarray(percent_passing, dtype=float)
    if np.any(np.diff(x) <= 0): raise ValueError("sizes_um must be strictly increasing.")
    if np.any(np.diff(y) < 0):  raise ValueError("percent_passing must be non-decreasing.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(x, y, "-o", linewidth=2, markersize=4)

    if x_limits: ax.set_xlim(*x_limits)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Particle size (\u03bcm)")  # your requested literal text
    ax.set_ylabel("Percent passing (%)")
    ax.set_title(title)

    ax.set_xticks(list(x_major_ticks))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    if show_grid:
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.2)

    dx_out: dict[str, float] = {}
    if annotate_dx:
        for p in annotate_dx:
            d = d_at_percent(x, y, p)
            dx_out[f"D{p}"] = d
            # (no lines/labels drawn on the figure)

    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return dx_out

# ---- All-samples (PSD_Full) ----
def _extract_numeric_size(colname: str) -> Optional[float]:
    txt = str(colname)
    # pull the first numeric token (handles '0.0995 u03bcm', '1110 u03bcm', etc.)
    for token in txt.split():
        try:
            return float(token)
        except ValueError:
            continue
    return None

def plot_all_psd(
    excel_path: str,
    sheet_name: str = "PSD_Full",
    *,
    title: str = "Particle Size Distribution (All Samples)",
    x_limits: tuple[float, float] = (0.1, 1000),
    x_major_ticks: Iterable[float] = (0.1, 1, 10, 100, 1000),
    show_grid: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    legend: bool = True,
):
    """
    Read PSD_Full and plot every row as a *cumulative* percent undersize curve.
    If the row already looks cumulative (monotonic non-decreasing <=100), it's used as-is.
    Otherwise it's treated as differential % per bin and converted to cumulative.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")

    # --- find numeric PSD columns (e.g., '0.0995 u03bcm', '1110 u03bcm') ---
    psd_cols: List[str] = []
    sizes_um: List[float] = []
    for c in df.columns:
        val = _extract_numeric_size(c)  # your helper stays the same
        if val is not None:
            psd_cols.append(c)
            sizes_um.append(val)

    if not psd_cols:
        raise ValueError("No numeric PSD size columns found in the sheet.")

    # sort by size (ascending) so cumulative is correct
    order = np.argsort(sizes_um)
    sizes_um = np.asarray(sizes_um, dtype=float)[order]
    psd_cols = [psd_cols[i] for i in order]

    def to_cumulative(y_raw: np.ndarray) -> np.ndarray:
        """Return cumulative % undersize (0-100). Detect if already cumulative."""
        y = y_raw.astype(float)
        # treat NaNs as zeros for processing
        y = np.nan_to_num(y, nan=0.0)
        # already cumulative? (non-decreasing and not >100)
        if np.all(np.diff(y) >= -1e-6) and np.nanmax(y) <= 100 + 1e-6:
            return np.clip(y, 0, 100)
        # assume differential -> normalise to 100 then cumsum
        total = y.sum()
        if total <= 0:
            return y  # all zeros
        y_norm = y * (100.0 / total)
        cum = np.cumsum(y_norm)
        return np.clip(cum, 0, 100)

    # --- plot all samples ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")

    for i, (_, row) in enumerate(df.iterrows()):
        name_val = row.get("Sample Name")
        if pd.isna(name_val):
            continue  # skip empty rows

        y_vals = row[psd_cols].to_numpy(dtype=float)
        if np.isnan(y_vals).all():
            continue  # skip rows with no PSD data

        y_cum = to_cumulative(y_vals)
        ax.semilogx(sizes_um, y_cum, linewidth=2, label=str(name_val), color=cmap(i % 20))

    ax.set_xlim(*x_limits)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Particle size (\u03bcm)")  # keep literal text as requested
    ax.set_ylabel("Cumulative percent undersize (%)")
    ax.set_title(title)

    ax.set_xticks(list(x_major_ticks))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    if show_grid:
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.2)

    if legend:
        ax.legend(title="Sample Code", fontsize=9, ncol=1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

