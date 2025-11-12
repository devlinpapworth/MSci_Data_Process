# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------
# Public convenience: consistent default colors for sample codes/names
# ---------------------------------------------------------------------
DEFAULT_SAMPLE_COLORS: Dict[str, str] = {
    "Si_F": "#1f77b4",       # blue
    "Si_M": "#ff7f0e",       # orange
    "Si_C": "#2ca02c",       # green
    "Si_Rep": "#d62728",     # red
    "Si_Rep_new": "#9467bd", # purple
    "RT_As Received": "#000000",
    "Si_val_01": "#1f77b4",
    "Si_val_02": "#ff7f0e",
    "Si_val_03": "#2ca02c",
    # add others as needed
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
    (Kept for reference; not used by the 'no interpolation' logic below.)
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
    annotate_dx: Iterable[int] | None = None,
    save_path: Optional[str] = None,
    show: bool = True,
    label: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
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
        curve_color = "#1f77b4"

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
# Helpers for extracting sizes and Dx without interpolation
# ---------------------------------------------------------------------
def _extract_numeric_size(colname: str) -> Optional[float]:
    for token in str(colname).split():
        try:
            return float(token)
        except ValueError:
            continue
    return None

def _first_size_at_or_above_percent(
    sizes: np.ndarray, cum: np.ndarray, target_percent: float
) -> float:
    """
    'No interpolation' rule: return the first size where cumulative >= target_percent.
    If never reached, return NaN.
    """
    idx = np.where(cum >= target_percent)[0]
    return float(sizes[idx[0]]) if idx.size else float("nan")

def _find_dx_column(df_row: pd.Series, targets=(10,20,80)) -> Dict[int, Optional[float]]:
    """
    Try to read Dx columns directly (e.g., 'Dx (20)', 'D10', 'Dx(80)').
    Returns dict {percentile: value or None}.
    """
    lookup = {p: None for p in targets}
    # Build a case-folded column index for robust matching
    cols = {c.casefold(): c for c in df_row.index.astype(str)}
    for p in targets:
        # possible names
        candidates = [
            f"dx ({p})", f"dx({p})", f"d{x}" if (x:=p) else "", f"d{p}",
            f"dx {p}", f"dx_{p}"
        ]
        for cand in candidates:
            key = cand.casefold()
            if key in cols:
                try:
                    val = float(df_row[cols[key]])
                    lookup[p] = val
                    break
                except Exception:
                    pass
    return lookup

# ---------------------------------------------------------------------
# All-samples (PSD_Full) with printed D10/D20/D80 + ratio (no interpolation)
# ---------------------------------------------------------------------
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
    include_samples: Optional[Iterable[Any]] = None,
    color_map: Optional[Dict[str, str]] = None,
    id_col_override: Optional[str] = None,
    print_dx: bool = True  # << NEW: print a table of D10/D20/D80/D80/20
) -> Optional[pandas.DataFrame]:

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

    # --- Filter rows if requested ---
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

    # collect Dx results
    rows_out: List[Dict[str, Any]] = []

    # plot each PSD curve
    for _, row in df.iterrows():
        label = str(row[id_col]).strip()
        y_vals = row[psd_cols].to_numpy(dtype=float)
        if np.isnan(y_vals).all():
            continue
        y_cum = to_cumulative(y_vals)

        # Plot curve
        ax.semilogx(sizes_um, y_cum, linewidth=2, label=label, color=resolve_color(label))

        # --- NO-INTERPOLATION Dx extraction ---
        # 1) Prefer explicit Dx columns if present
        dx_cols = _find_dx_column(row, targets=(10, 20, 80))

        # 2) Fallback to first bin meeting the target (>=) from cumulative curve
        d10 = dx_cols[10] if dx_cols[10] is not None else _first_size_at_or_above_percent(sizes_um, y_cum, 10.0)
        d20 = dx_cols[20] if dx_cols[20] is not None else _first_size_at_or_above_percent(sizes_um, y_cum, 20.0)
        d80 = dx_cols[80] if dx_cols[80] is not None else _first_size_at_or_above_percent(sizes_um, y_cum, 80.0)

        span = (d80 / d20) if (np.isfinite(d80) and np.isfinite(d20) and d20 > 0) else np.nan

        rows_out.append({
            id_col: label,
            "D10 (um)": d10,
            "D20 (um)": d20,
            "D80 (um)": d80,
            "D80/D20": span,
        })

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

    # --- Print Dx table (no interpolation) ---
    if print_dx and rows_out:
        out_df = pd.DataFrame(rows_out)
        # order columns nicely
        out_df = out_df[[id_col, "D10 (um)", "D20 (um)", "D80 (um)", "D80/D20"]]
        # round for display
        print(out_df.round({"D10 (um)": 2, "D20 (um)": 2, "D80 (um)": 2, "D80/D20": 2}).to_string(index=False))
        return out_df

    return None
