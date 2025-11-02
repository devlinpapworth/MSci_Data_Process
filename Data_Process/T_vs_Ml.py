# Data_Process/T_vs_Ml.py
# Plot F_T (x) vs F_T/F_V (y), color-coded by Sample Code, with same filters as other funcs.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_T_vs_Ml(xlsx_path, sheet_db="DB"):
    """
    x = F_T
    y = F_T / F_V
    Filters out rows where 'Coments' contains 'fail' or 'anom'.
    Colors by Sample Code. Returns filtered DataFrame with 'FT_over_FV'.
    """
    # === Load ===
    df = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")

    # (Optional) very light header cleanup without changing semantics
    # df.columns = df.columns.str.strip()

    # === Required columns present? ===
    required = ["F_T", "F_V", "Sample Code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in '{sheet_db}': {missing}")

    # === Coerce numerics ===
    df["F_T"] = pd.to_numeric(df["F_T"], errors="coerce")
    df["F_V"] = pd.to_numeric(df["F_V"], errors="coerce")

    # === Filter 'fail'/'anom' ===
    if "flag" in df.columns:
        mask_bad = df["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df = df[~mask_bad]

    # === Valid rows ===
    df = df.dropna(subset=["Sample Code", "F_T", "F_V"])
    df = df[df["F_V"] != 0]

    # === Compute y ===
    df["FT_over_FV"] = df["F_T"] / df["F_V"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FT_over_FV"])

    # === Labels & stable colors ===
    df["Sample Label"] = df["Sample Code"].astype(str)
    groups = df["Sample Label"]
    uniq = sorted(groups.unique())
    cmap = plt.cm.get_cmap("tab20", max(1, len(uniq)))
    color_map = {g: cmap(i % cmap.N) for i, g in enumerate(uniq)}

    # === Plot ===
    plt.figure(figsize=(8.0, 5.8))
    for g in uniq:
        m = (groups == g)
        plt.scatter(df.loc[m, "F_T"], df.loc[m, "FT_over_FV"],
                    s=65, alpha=0.9, edgecolor="black", linewidths=0.4,
                    color=color_map[g], label=g)

    # Trendline (overall)
    try:
        x = df["F_T"].values
        y = df["FT_over_FV"].values
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            plt.plot(xs, p(xs), linestyle="--", color="black", label="Trend")
    except Exception:
        pass

    plt.title("Time vs (Time / Volume Filtrate)")
    plt.xlabel("Time (F_T)")
    plt.ylabel("Time / Volume Filtrate (F_T / F_V)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Sample Code", fontsize=8, bbox_to_anchor=(1.02, 1),
               loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    return df

def plot_TvsPSD(xlsx_path, sheet_db="DB", sheet_psd="PSD"):
    """
    x = D90 / D10 (EFI)
    y = F_T / F_V
    Filters 'fail'/'anom'. Colors by Sample Code.
    Returns merged DataFrame with 'FT_over_FV' and 'EFI'.
    """
    # === Load ===
    df_db = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # === Required columns ===
    required_db = ["F_T", "F_V", "Sample Code"]
    required_psd = ["D90", "D50"]
    miss_db = [c for c in required_db if c not in df_db.columns]
    miss_psd = [c for c in required_psd if c not in df_psd.columns]
    if miss_db:
        raise ValueError(f"Missing in '{sheet_db}': {miss_db}")
    if miss_psd:
        raise ValueError(f"Missing in '{sheet_psd}': {miss_psd}")

    # === Coerce numerics ===
    df_db["F_T"] = pd.to_numeric(df_db["F_T"], errors="coerce")
    df_db["F_V"] = pd.to_numeric(df_db["F_V"], errors="coerce")
    df_psd["D90"] = pd.to_numeric(df_psd["D90"], errors="coerce")
    df_psd["D50"] = pd.to_numeric(df_psd["D50"], errors="coerce")

    # === Filter 'fail'/'anom' ===
    if "flag" in df_db.columns:
        mask_bad = df_db["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df_db = df_db[~mask_bad]

    # === Merge ===
    df = pd.merge(df_db, df_psd, on="Sample Code", how="inner")

    # === Keep valid rows ===
    df = df.dropna(subset=["Sample Code", "F_T", "F_V", "D90", "D50"])
    df = df[(df["F_V"] != 0) & (df["D90"] > 0) & (df["D50"] > 0)]

    # === Compute axes ===
    df["FT_over_FV"] = df["F_T"] / df["F_V"]
    df["EFI"] = df["D90"] / df["D50"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FT_over_FV", "EFI"])

    # === Labels & stable colors ===
    df["Sample Label"] = df["Sample Code"].astype(str)
    groups = df["Sample Label"]
    uniq = sorted(groups.unique())
    cmap = plt.cm.get_cmap("tab20", max(1, len(uniq)))
    color_map = {g: cmap(i % cmap.N) for i, g in enumerate(uniq)}

    # === Plot ===
    plt.figure(figsize=(8.0, 5.8))
    for g in uniq:
        m = (groups == g)
        plt.scatter(df.loc[m, "EFI"], df.loc[m, "FT_over_FV"],
                    s=65, alpha=0.9, edgecolor="black", linewidths=0.4,
                    color=color_map[g], label=g)

    # Trendline (overall)
    try:
        x = df["EFI"].values
        y = df["FT_over_FV"].values
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            plt.plot(xs, p(xs), linestyle="--", color="black", label="Trend")
    except Exception:
        pass

    plt.title("Time / Volume Filtrate vs (D90 / D50)")
    plt.xlabel("D90 / D10")
    plt.ylabel("Time / Volume Filtrate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Sample Code", fontsize=8, bbox_to_anchor=(1.02, 1),
               loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    return df
