import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_airp_vs_psd(
    xlsx_path,
    sheet_db="DB",
    sheet_psd="PSD",
    include_samples=None,
    color_map=None,   # dict: sample -> color
):
    

    # === Load ===
    df_db  = pd.read_excel(xlsx_path, sheet_name=sheet_db,  engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # === Coerce needed columns ===
    for col in ["A_P", "A_T", "A_V"]:
        if col not in df_db.columns:
            raise ValueError(f"'{col}' is missing from sheet '{sheet_db}'.")
        df_db[col] = pd.to_numeric(df_db[col], errors="coerce")

    # Keep only rows explicitly marked 'include'
    if "flag" in df_db.columns:
        keep = df_db["flag"].astype(str).str.lower().str.contains(r"\binclude\b", na=False)
        df_db = df_db[keep]

    # Optional manual include filter
    if include_samples is not None:
        allow = set(str(s) for s in include_samples)
        df_db = df_db[df_db["Sample Code"].astype(str).isin(allow)]

    # PSD columns
    for col in ["D10", "D50", "D90"]:
        if col not in df_psd.columns:
            raise ValueError(f"'{col}' is missing from sheet '{sheet_psd}'.")
        df_psd[col] = pd.to_numeric(df_psd[col], errors="coerce")

    # Merge
    df = pd.merge(df_db, df_psd, on="Sample Code", how="inner")

    # Require D10 present and positive
    df = df.dropna(subset=["Sample Code", "D10", "A_P"])
    df = df[df["D10"] > 0]

    # Ratios
    df["A_P_over_A_T"] = np.where(df["A_T"].notna() & (df["A_T"] != 0), df["A_P"] / df["A_T"], np.nan)
    df["A_P_over_A_V"] = np.where(df["A_V"].notna() & (df["A_V"] != 0), df["A_P"] / df["A_V"], np.nan)

    # Labels/colors
    df["Sample Label"] = df["Sample Code"].astype(str)

    # Robust ylim helper (2–98th pct + padding)
    def robust_ylim(y):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size < 2:
            return None
        lo, hi = np.percentile(y, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return None
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        return (lo - pad, hi + pad)

    # Generic scatter helper
    def scatter_grouped(x, y, groups, xlab, ylab, title, xlim=None, ylim_auto=True):
        plt.figure(figsize=(8.0, 5.8))
        uniq = sorted(groups.unique())

        # color lookup
        if color_map is not None:
            lookup = {g: color_map.get(g, "#808080") for g in uniq}
        else:
            cmap = plt.cm.get_cmap("tab20", len(uniq))
            lookup = {g: cmap(i) for i, g in enumerate(uniq)}

        for g in uniq:
            m = (groups == g) & np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                continue
            plt.scatter(x[m], y[m], s=65, alpha=0.9, edgecolor="black",
                        linewidths=0.4, color=lookup[g], label=g)

        # Trendline
        try:
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
                plt.plot(xs, p(xs), linestyle="--", color="black", label="Trend")
        except Exception:
            pass

        # Axes
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim_auto:
            yl = robust_ylim(y)
            if yl is not None:
                plt.ylim(*yl)

        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Sample Code", fontsize=8, bbox_to_anchor=(1.02, 1),
                   loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    # --- Plot 1: A_P vs D10 --- 
    scatter_grouped(
        df["D10"].to_numpy(), df["A_P"].to_numpy(), df["Sample Label"],
        xlab="D10 (\u03bcm)", ylab="A_P (bar)", title="Air-Blow Pressure vs D10",
        xlim=(0, 100), ylim_auto=True
    )

    # --- Plot 2: A_P / A_T vs D10 ---
    scatter_grouped(
        df["D10"].to_numpy(), df["A_P_over_A_T"].to_numpy(), df["Sample Label"],
        xlab="D10 (\u03bcm)", ylab="A_P / A_T (bar/s)", title="Air-Blow Pressure / Time vs D10",
        xlim=(0, 100), ylim_auto=True
    )

    # --- Plot 3: A_P / A_V vs D10 ---
    scatter_grouped(
        df["D10"].to_numpy(), df["A_P_over_A_V"].to_numpy(), df["Sample Label"],
        xlab="D10 (\u03bcm)", ylab="A_P / A_V (bar/ml)", title="Air-Blow Pressure / Volume vs D10",
        xlim=(0, 100), ylim_auto=True
    )

    return df
