# Data_Process/T_vs_Ml.py
# Plot F_T (x) vs F_T/F_V (y), color-coded by Sample Code, with same filters as other funcs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_T_vs_Ml(xlsx_path, sheet_db="DB"):
    """
    Reads the DB sheet and plots:
        x = F_T
        y = F_T / F_V
    Excludes rows where 'Coments' contains 'fail' or 'anom' (case-insensitive).
    Color-codes points by Sample Code. Adds an overall linear trendline.
    Returns the filtered DataFrame with computed column 'FT_over_FV'.
    """

    # === Load data ===
    df = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")

    # === Coerce needed columns to numeric ===
    for col in ["F_T", "F_V", "Sample Code"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in sheet '{sheet_db}'.")

    df["F_T"] = pd.to_numeric(df.get("F_T"), errors="coerce")
    df["F_V"] = pd.to_numeric(df.get("F_V"), errors="coerce")

    # === Drop failed/anom rows if 'Coments' exists ===
    if "Coments" in df.columns:
        bad = df["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df = df[~bad]

    # === Keep only valid rows ===
    df = df.dropna(subset=["Sample Code", "F_T", "F_V"])
    df = df[df["F_V"] != 0]  # avoid division by zero

    # === Compute y = F_T / F_V ===
    df["FT_over_FV"] = df["F_T"] / df["F_V"]

    # Drop inf / NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FT_over_FV"])

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

        # Overall trendline (optional)
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
        df["F_T"].values, df["FT_over_FV"].values, df["Sample Label"],
        xlab="F_T", ylab="F_T / F_V",
        title="F_T vs (F_T / F_V)"
    )

    return df
def plot_TvsPSD(xlsx_path, sheet_db="DB", sheet_psd="PSD"):
    """
    Reads the DB and PSD sheets and plots:
        x = D50 / D10  (EFI)
        y = F_T / F_V
    Excludes rows where 'Coments' contains 'fail' or 'anom' (case-insensitive).
    Color-codes points by Sample Code. Adds an overall linear trendline.
    Returns the filtered DataFrame with computed columns 'FT_over_FV' and 'EFI'.
    """

    # === Load data ===
    df_db = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # === Check required columns ===
    for col in ["F_T", "F_V", "Sample Code"]:
        if col not in df_db.columns:
            raise ValueError(f"Required column '{col}' not found in sheet '{sheet_db}'.")
    for col in ["D10", "D50"]:
        if col not in df_psd.columns:
            raise ValueError(f"Required column '{col}' not found in sheet '{sheet_psd}'.")

    # === Coerce numeric ===
    df_db["F_T"] = pd.to_numeric(df_db.get("F_T"), errors="coerce")
    df_db["F_V"] = pd.to_numeric(df_db.get("F_V"), errors="coerce")
    df_psd["D10"] = pd.to_numeric(df_psd.get("D10"), errors="coerce")
    df_psd["D50"] = pd.to_numeric(df_psd.get("D50"), errors="coerce")

    # === Drop failed/anom rows if 'Coments' exists ===
    if "Coments" in df_db.columns:
        bad = df_db["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df_db = df_db[~bad]

    # === Merge PSD data by Sample Code ===
    df = pd.merge(df_db, df_psd, on="Sample Code", how="inner")

    # === Keep valid rows ===
    df = df.dropna(subset=["Sample Code", "F_T", "F_V", "D10", "D50"])
    df = df[(df["F_V"] != 0) & (df["D10"] > 0) & (df["D50"] > 0)]

    # === Compute y = F_T / F_V, x = D50 / D10 ===
    df["FT_over_FV"] = df["F_T"] / df["F_V"]
    df["EFI"] = df["D50"] / df["D10"]  # Effective Fine Index

    # Drop inf / NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FT_over_FV", "EFI"])

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

        # Overall trendline (optional)
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
        df["EFI"].values, df["FT_over_FV"].values, df["Sample Label"],
        xlab="EFI = D50 / D10 (-)",
        ylab="F_T / F_V",
        title="F_T / F_V vs EFI (D50 / D10)"
    )

    return df