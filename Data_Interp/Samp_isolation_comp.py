# Data_Interp/Samp_isolation_comp.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from typing import Optional, Dict, List


def plot_moisture_category_bars(
    xlsx_path: str,
    sheet_db: str = "DB",
    *,
    color_map: Optional[Dict[str, str]] = None,
    only_flag_include: bool = False,
    annotate: bool = True,
) -> pd.DataFrame:
    

    # --- Category definitions ---
    categories: List[tuple[str, List[str]]] = [
        ("isolated slope below D50", ["Si_F", "Si_Rep_new"]),
        ("isolated slope", ["Si_C", "Si_M"]),
        ("isolated width/end members", ["Si_BM", "Si_Rep_new"]),
        ("isolated slope above D50", ["Si_BM", "Si_Rep"]),
    ]

    # --- Load and clean DB ---
    df = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")

    if "Sample Code" not in df.columns or "Mc_%" not in df.columns:
        raise ValueError("DB sheet must contain 'Sample Code' and 'Mc_%' columns.")

    df["Mc_%"] = pd.to_numeric(df["Mc_%"], errors="coerce")

    if only_flag_include and "flag" in df.columns:
        df = df[df["flag"].astype(str).str.lower().eq("include")]

    if "Coments" in df.columns:
        bad = df["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df = df[~bad]

    df = df.dropna(subset=["Sample Code", "Mc_%"])

    # Convert to % if stored as 0–1
    if df["Mc_%"].max() <= 1.5:
        df["Mc_%"] = df["Mc_%"] * 100.0

    # --- Collect sample codes used ---
    needed_codes = sorted({code for _, pair in categories for code in pair})

    # --- Colours ---
    if color_map is None:
        cmap = plt.get_cmap("tab10", max(2, len(needed_codes)))
        color_map = {code: cmap(i % cmap.N) for i, code in enumerate(needed_codes)}
    else:
        for code in needed_codes:
            color_map.setdefault(code, "grey")

    # --- Prepare data for plotting ---
    cats = [c[0] for c in categories]
    x_positions = np.arange(len(cats), dtype=float)
    offset = 0.15  # horizontal jitter spacing between the two sample codes in each category

    fig, ax = plt.subplots(figsize=(10, 5.8))

    # --- Plot each category ---
    rows = []  # for returned stats
    for i, (cat_name, codes) in enumerate(categories):
        # space left/right for two codes
        sub_offsets = np.linspace(-offset, offset, len(codes))
        for j, code in enumerate(codes):
            y_vals = df.loc[df["Sample Code"].astype(str) == code, "Mc_%"].to_numpy()
            y_vals = y_vals[np.isfinite(y_vals)]

            # scatter each data point with jitter
            ax.scatter(
                np.full_like(y_vals, x_positions[i] + sub_offsets[j]),
                y_vals,
                color=color_map.get(code, "gray"),
                edgecolor="black",
                s=60,
                alpha=0.8,
                label=code if i == 0 else None  # label only first instance per code for legend
            )

            # save summary stats
            if y_vals.size:
                rows.append({
                    "Category": cat_name,
                    "Sample Code": code,
                    "n": int(y_vals.size),
                    "mean": float(np.nanmean(y_vals)),
                    "median": float(np.nanmedian(y_vals)),
                    "std": float(np.nanstd(y_vals, ddof=1)) if y_vals.size > 1 else np.nan,
                })

            # annotate n above mean
            if annotate and y_vals.size:
                ax.text(
                    x_positions[i] + sub_offsets[j],
                    np.nanmax(y_vals) + 0.3,
                    f"n={y_vals.size}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # --- Axes and styling ---
    ax.set_xticks(x_positions, cats)
    ax.set_ylabel("Final Moisture (Mc %)")
    ax.set_title("Final Moisture by Category")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # fixed y-axis 0–30 %
    ax.set_ylim(0, 30)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f} %"))

    # --- Legend (one entry per sample code) ---
    proxies = [Patch(facecolor=color_map[c], edgecolor="black", label=c) for c in needed_codes]
    ax.legend(handles=proxies, title="Sample Code", fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    plt.show()

    # --- Return summary stats ---
    return pd.DataFrame(rows)
