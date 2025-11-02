# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import Iterable, Optional, Sequence, Tuple, List

# ---------- Single-curve helpers ----------
def d_at_percent(
    sizes_um: Sequence[float],
    percent_passing: Sequence[float],
    p: float
) -> float:
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
) -> dict[str, float]:
    x = np.asarray(sizes_um, dtype=float)
    y = np.asarray(percent_passing, dtype=float)
    if np.any(np.diff(x) <= 0): raise ValueError("sizes_um must be strictly increasing.")
    if np.any(np.diff(y) < 0):  raise ValueError("percent_passing must be non-decreasing.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(x, y, "-o", linewidth=2, markersize=4)
    if x_limits: ax.set_xlim(*x_limits)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Particle size (\u03bcm)")
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
            # (no guide lines or text drawn)

    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return dx_out

# ---------- All-samples (PSD_Full) ----------
def _extract_numeric_size(colname: str) -> Optional[float]:
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
    title: str = "Particle Size Distribution (All Samples)",
    x_limits: Tuple[float, float] = (0.1, 1000),
    x_major_ticks: Iterable[float] = (0.1, 1, 10, 100, 1000),
    show_grid: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    legend: bool = True,
):
    def to_cumulative(y_raw: np.ndarray) -> np.ndarray:
        y = np.nan_to_num(y_raw.astype(float), nan=0.0)
        if np.all(np.diff(y) >= -1e-6) and np.nanmax(y) <= 100 + 1e-6:
            return np.clip(y, 0, 100)
        tot = y.sum()
        return np.cumsum(y * (100.0 / tot)) if tot > 0 else y

    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")

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

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    cidx = 0
    for _, row in df.iterrows():
        name_val = row.get("Sample Name")
        if pd.isna(name_val):
            continue
        y_vals = row[psd_cols].to_numpy(dtype=float)
        if np.isnan(y_vals).all():
            continue
        y_cum = to_cumulative(y_vals)
        ax.semilogx(sizes_um, y_cum, linewidth=2, label=str(name_val), color=cmap(cidx % 20))
        cidx += 1

    # Overlay your manual curve automatically
    manual_sizes = np.array([0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 26, 38, 53, 75, 106, 150, 212, 300, 425, 600, 1000], dtype=float)
    manual_percent = np.array([2.826, 4.277, 7.206, 12.586, 17.421, 25.123, 31.079, 40.088, 46.615,
                               55.289, 62.971, 69.272, 75.268, 80.019, 84.453, 88.355, 91.703,
                               94.637, 97.28, 99.165, 99.914, 100], dtype=float)
    mo = np.argsort(manual_sizes)
    manual_sizes = manual_sizes[mo]
    manual_percent = manual_percent[mo]
    manual_cum = to_cumulative(manual_percent)
    ax.semilogx(manual_sizes, manual_cum, linewidth=3, color="black", label="RT_As Received")

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
        ax.legend(title="Sample Code", fontsize=9, ncol=1)

    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
