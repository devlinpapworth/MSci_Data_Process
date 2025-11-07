# Data_Interp/Samp_isolation_comp.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib import transforms as mtransforms
from typing import Optional, Dict


def plot_FT_over_FV_violins_by_code(
    xlsx_path: str,
    sheet_db: str = "DB",
    *,
    color_map: Optional[Dict[str, str]] = None,
    annotate: bool = True,
    sort_codes: str = "alpha",   # "alpha" | "n" | "median" | "mean"
    show_points: bool = True,    # overlay individual observations
) -> pd.DataFrame:
    """
    Plot FT/FV (filling time over filling volume) as violin plots for all Sample Codes
    flagged as 'include'. Marker shapes denote Test_procedure (STD, No_pres, Other).
    Returns summary statistics per code.
    """

    # --- Load + basic checks ---
    df = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")
    if "Sample Code" not in df.columns or not {"F_T", "F_V"} <= set(df.columns):
        raise ValueError("DB sheet must contain 'Sample Code', 'F_T', and 'F_V' columns.")

    # Convert numeric columns
    df["F_T"] = pd.to_numeric(df["F_T"], errors="coerce")
    df["F_V"] = pd.to_numeric(df["F_V"], errors="coerce")

    # keep only flag == include
    if "flag" in df.columns:
        df = df[df["flag"].astype(str).str.lower().eq("include")]

    # drop fails/anoms
    if "Coments" in df.columns:
        bad = df["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df = df[~bad]

    # --- Normalize Test_procedure labels ---
    if "Test_procedure" in df.columns:
        tp = (
            df["Test_procedure"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[\s\-]+", "_", regex=True)
        )
        df["TP_norm"] = np.where(
            tp.eq("std"),
            "STD",
            np.where(tp.isin({"no_pres", "no_press", "nopress"}), "No_pres", "Other"),
        )
    else:
        df["TP_norm"] = "Other"

    # --- Compute FT/FV ratio ---
    df = df.dropna(subset=["Sample Code", "F_T", "F_V"])
    df = df[df["F_V"] != 0]
    df["FT_over_FV"] = df["F_T"] / df["F_V"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FT_over_FV"])

    # --- Group for violins ---
    by_code = (
        df.groupby("Sample Code")["FT_over_FV"]
        .apply(lambda s: s.dropna().to_numpy())
        .to_dict()
    )
    by_code = {k: v for k, v in by_code.items() if len(v) > 0}

    # --- Order on x ---
    stats_temp = [
        {
            "Sample Code": c,
            "n": int(len(v)),
            "mean": float(np.nanmean(v)),
            "median": float(np.nanmedian(v)),
        }
        for c, v in by_code.items()
    ]
    order_df = pd.DataFrame(stats_temp)

    if sort_codes == "n":
        order_df = order_df.sort_values(["n", "Sample Code"], ascending=[False, True])
    elif sort_codes == "median":
        order_df = order_df.sort_values(["median", "Sample Code"])
    elif sort_codes == "mean":
        order_df = order_df.sort_values(["mean", "Sample Code"])
    else:
        order_df = order_df.sort_values("Sample Code")

    codes = order_df["Sample Code"].tolist()
    datasets = [by_code[c] for c in codes]

    # --- Colours ---
    if color_map is None:
        cmap = plt.get_cmap("tab10", max(2, len(codes)))
        color_map = {code: cmap(i % cmap.N) for i, code in enumerate(codes)}
    else:
        for code in codes:
            color_map.setdefault(code, "grey")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(codes))

    # Violin plot
    parts = ax.violinplot(
        datasets,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.8,
        bw_method=0.3,
        points=200,
    )

    # Scale violin width by replicate count
    ns = np.array([len(v) for v in datasets], dtype=float)
    max_n = max(1.0, ns.max())
    width_scales = 0.35 + 0.65 * np.sqrt(ns / max_n)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(color_map[codes[i]])
        body.set_edgecolor("black")
        body.set_alpha(0.75)
        body.set_linewidth(0.9)

        # Scale width
        sx = width_scales[i]
        t = (
            mtransforms.Affine2D()
            .translate(-positions[i], 0)
            .scale(sx, 1.0)
            .translate(positions[i], 0)
            + ax.transData
        )
        body.set_transform(t)

        # Clip KDE range
        y_min = float(np.min(datasets[i]))
        y_max = float(np.max(datasets[i]))
        path = body.get_paths()[0]
        verts = path.vertices
        verts[:, 1] = np.clip(verts[:, 1], y_min, y_max)
        path.vertices = verts

    if "cmedians" in parts and parts["cmedians"] is not None:
        parts["cmedians"].set_linewidth(2.0)

    # --- Quartile lines ---
    for i, vals in enumerate(datasets):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([positions[i] - 0.18, positions[i] + 0.18], [q1, q1], lw=1, color="black", alpha=0.7)
        ax.plot([positions[i] - 0.22, positions[i] + 0.22], [q3, q3], lw=1, color="black", alpha=0.7)

    # --- Overlay points ---
    marker_for_tp = {"STD": "o", "No_pres": "^", "Other": "s"}
    if show_points:
        rng = np.random.default_rng(42)
        for i, code in enumerate(codes):
            sub = df[df["Sample Code"].astype(str) == code]
            if sub.empty:
                continue
            xj = positions[i] + rng.normal(0, 0.06, size=len(sub))
            for (xpos, y, tp_label) in zip(xj, sub["FT_over_FV"].to_numpy(), sub["TP_norm"].to_numpy()):
                ax.scatter(
                    xpos, y,
                    s=32,
                    edgecolor="black",
                    facecolor=color_map[code],
                    marker=marker_for_tp.get(tp_label, "s"),
                    alpha=0.9,
                    linewidth=0.6,
                    zorder=3,
                )

    # --- Stats + annotations ---
    rows = []
    for i, (code, vals) in enumerate(zip(codes, datasets)):
        n = len(vals)
        mean = np.nanmean(vals)
        median = np.nanmedian(vals)
        std = np.nanstd(vals, ddof=1) if n > 1 else np.nan
        rows.append({"Sample Code": code, "n": n, "mean": mean, "median": median, "std": std})
        if annotate and n:
            ax.text(
                positions[i],
                np.nanmax(vals) * 1.02,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # --- Axes & styling ---
    ax.set_xticks(positions, codes, rotation=30, ha="right")
    ax.set_ylabel(r"Time / Volume Filtrate  (s/ml)")
    ax.set_title("Filling Time / Volume Ratio by Sample Code")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

    # --- Legend (combined) ---
    fig.subplots_adjust(right=0.83)

    color_handles = [Patch(facecolor=color_map[c], edgecolor="black", label=c) for c in codes]
    marker_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markeredgecolor='black',
               markerfacecolor='white', markersize=8, label='STD'),
        Line2D([0], [0], marker='^', linestyle='None', markeredgecolor='black',
               markerfacecolor='white', markersize=8, label='No_pres'),
        Line2D([0], [0], marker='s', linestyle='None', markeredgecolor='black',
               markerfacecolor='white', markersize=8, label='Other'),
    ]
    header_sample = Line2D([], [], linestyle='none', label='Sample Code')
    header_tp     = Line2D([], [], linestyle='none', label='Test procedure')
    combined_handles = [header_sample] + color_handles + [header_tp] + marker_handles

    leg = fig.legend(
        handles=combined_handles,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        fontsize=9,
        frameon=True,
        borderaxespad=0.0,
        handlelength=1.4,
    )
    for txt in leg.get_texts():
        if txt.get_text() in ("Sample Code", "Test procedure"):
            txt.set_fontweight("bold")

    plt.show()

    return pd.DataFrame(rows)
