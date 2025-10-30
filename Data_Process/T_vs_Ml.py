
# Data_Process/T_vs_Ml.py
# Plot F_T (x) vs F_V/F_T (y), color-coded by Sample Code, with same filters as other funcs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_T_vs_Ml(xlsx_path, sheet_db="DB"):
    """
    Reads the DB sheet and plots:
        x = F_T
        y = F_V / F_T
    Excludes rows where 'Coments' contains 'fail' or 'anom' (case-insensitive).
    Color-codes points by Sample Code. Adds an overall linear trendline.
    Returns the filtered DataFrame with computed column 'FV_over_FT'.
    """

    # === Load data ===
    df = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")

    # === Coerce needed columns to numeric ===
    # Use .get to avoid KeyError if a column is missing; we will check later.
    for col in ["F_T", "F_V", "Sample Code"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in sheet '{sheet_db}'.")

    df["F_T"] = pd.to_numeric(df.get("F_T"), errors="coerce")
    df["F_V"] = pd.to_numeric(df.get("F_V"), errors="coerce")

    # Optional moisture if you later want to reuse; not required here.
    # df["Mc_%"] = pd.to_numeric(df.get("Mc_%"), errors="coerce")

    # === Drop failed/anom rows if 'Coments' exists ===
    if "Coments" in df.columns:
        bad = df["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df = df[~bad]

    # === Keep only valid rows ===
    # Need Sample Code, F_T, F_V; also avoid division by zero.
    df = df.dropna(subset=["Sample Code", "F_T", "F_V"])
    df = df[df["F_T"] != 0]

    # === Compute y = F_V / F_T ===
    df["FV_over_FT"] = df["F_V"] / df["F_T"]

    # Drop inf / NaN that may arise from weird entries
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FV_over_FT"])

    # === Prepare labels/groups ===
    df["Sample Label"] = df["Sample Code"].astype(str)

    # === Plot helper ===
    def scatter_grouped(x, y, groups, xlab, ylab, title):
        plt.figure(figsize=(8.0, 5.8))

        uniq = sorted(groups.unique())
        cmap = plt.cm.get_cmap("tab20", len(uniq))
        color_map = {g: cmap(i) for i, g in enumerate(uniq)}

        for g in uniq:
            m = (groups == g)
            plt.scatter(x[m], y[m], s=65, alpha=0.9, edgecolor="black",
                        linewidths=0.4, color=color_map[g], label=g)

        # Overall trendline (try; skip if not enough points)
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            plt.plot(xs, p(xs), linestyle="--", color="black", label="Trend")
        except Exception:
            pass

        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Sample Code", fontsize=8, bbox_to_anchor=(1.02, 1),
                   loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    # === Make the plot ===
    scatter_grouped(
        df["F_T"].values, df["FV_over_FT"].values, df["Sample Label"],
        xlab="F_T", ylab="F_V / F_T",
        title="F_T vs (F_V / F_T)"
    )

    return df
