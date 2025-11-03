import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_moisture_vs_psd_indices(
    xlsx_path,
    sheet_db="DB",
    sheet_psd="PSD",
    include_samples=None,
    color_map=None,   # pass dict of sample->color
):
    """
    Plot every valid row from DB (no aggregation) merged with PSD.
    Keeps only rows where 'flag' contains 'include' (case-insensitive).
    Optional manual filter via include_samples (iterable of codes).
    Color-code points by Sample Code. Shows four plots with FIXED AXES:
      1) Mc% vs D90/D50   (x: 1–10,   y: 0–30)
      2) Mc% vs FSI       (x: 0–5*,   y: 0–30)
      3) Mc% vs D10       (x: 0–100,  y: 0–30)
      4) Mc% vs D50       (x: 0–130,  y: 0–30)
    """

    # === Load data ===
    df_db  = pd.read_excel(xlsx_path, sheet_name=sheet_db,  engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # === Clean moisture ===
    df_db["Mc_%"] = pd.to_numeric(df_db.get("Mc_%"), errors="coerce")

    # === Convert decimals to percentage if needed ===
    if df_db["Mc_%"].mean(skipna=True) < 1:  # detects if values like 0.15 should be 15%
        df_db["Mc_%"] *= 100

    # === Keep only rows explicitly marked 'include' under the 'flag' column ===
    if "flag" in df_db.columns:
        keep = df_db["flag"].astype(str).str.lower().str.contains(r"\binclude\b", na=False)
        df_db = df_db[keep]

    # === Keep only rows with a sample code and valid moisture ===
    df_db = df_db.dropna(subset=["Sample Code", "Mc_%"])

    # === Manual include filter (apply before merge) ===
    if include_samples is not None:
        allow = set(str(s) for s in include_samples)
        df_db = df_db[df_db["Sample Code"].astype(str).isin(allow)]

    # === Prepare PSD table ===
    for col in ["D10", "D50", "D90"]:
        df_psd[col] = pd.to_numeric(df_psd.get(col), errors="coerce")

    # === Merge DB rows with PSD by Sample Code (no aggregation) ===
    df = pd.merge(df_db, df_psd, on="Sample Code", how="inner")

    # === Require PSD and Mc_% present and positive D10/D50 ===
    df = df.dropna(subset=["D10", "D50", "D90", "Mc_%"])
    df = df[(df["D10"] > 0) & (df["D50"] > 0)]

    # === Compute indices per-row ===
    df["D90_over_D50"] = df["D90"] / df["D50"]                        # Plot 1 x-axis
    df["FSI"]          = (df["D90"] - df["D10"]) / df["D50"]          # dimensionless span

    # === Color by exact Sample Code ===
    df["Sample Label"] = df["Sample Code"].astype(str)
    labels_series = df["Sample Label"]

    # === Generic scatter helper with color groups + FIXED AXES ===
    def scatter_grouped(x, y, groups, all_labels, xlab, ylab, title, xlim=None, ylim=None):
        plt.figure(figsize=(8.0, 5.8))

        uniq = sorted(groups.unique())

        # Use provided color_map if given; fallback to tab20; default grey for missing keys
        if color_map is not None:
            lookup = {g: color_map.get(g, "#808080") for g in uniq}
        else:
            cmap = plt.cm.get_cmap("tab20", len(uniq))
            lookup = {g: cmap(i) for i, g in enumerate(uniq)}

        for g in uniq:
            m = (groups == g)
            plt.scatter(x[m], y[m], s=65, alpha=0.9, edgecolor="black",
                        linewidths=0.4, color=lookup[g], label=g)

        # Trendline
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if mask.sum() >= 2:
                z = np.polyfit(x_arr[mask], y_arr[mask], 1)
                p = np.poly1d(z)
                xs = np.linspace(np.nanmin(x_arr[mask]), np.nanmax(x_arr[mask]), 200)
                plt.plot(xs, p(xs), linestyle="--", color="black", label="Trend")
        except Exception:
            pass

        # >>> FIXED AXES <<<
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)

        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Sample Code", fontsize=8, bbox_to_anchor=(1.02, 1),
                   loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    # === Plot 1: D90/D50 vs Mc_% ===
    scatter_grouped(
        df["D90_over_D50"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values,
        xlab="D90 / D50", ylab="Final Moisture (Mc %)",
        title="Final Moisture vs D90/D50",
        xlim=(1, 10), ylim=(0, 30)
    )

    # === Plot 2: FSI vs Mc_% ===
    scatter_grouped(
        df["FSI"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values,
        xlab="(D90 - D10) / D50", ylab="Final Moisture (Mc %)",
        title="Final Moisture vs (D90 - D10) / D50",
        xlim=(0, 9), ylim=(0, 30)
    )

    # === Plot 3: D10 vs Mc_% ===
    scatter_grouped(
        df["D10"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values,
        xlab="D10 (\u03bcm)", ylab="Final Moisture (Mc %)",
        title="Final Moisture vs D10",
        xlim=(0, 100), ylim=(0, 30)
    )

    # === Plot 4: D50 vs Mc_% ===
    scatter_grouped(
        df["D50"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values,
        xlab="D50 (\u03bcm)", ylab="Final Moisture (Mc %)",
        title="Final Moisture vs D50",
        xlim=(0, 130), ylim=(0, 30)
    )

    return df
