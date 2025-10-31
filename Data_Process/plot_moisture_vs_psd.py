import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_moisture_vs_psd_indices(xlsx_path, sheet_db="DB", sheet_psd="PSD"):
    """
    Plot every valid row from DB (no aggregation) merged with PSD.
    Excludes rows where 'Coments' contains 'fail' or 'anom' (case-insensitive).
    Color-code points by Sample Code. Shows three plots:
      - Mc% vs EFI
      - Mc% vs FSI (span)
      - Mc% vs D10
    """

    # === Load data ===
    df_db = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # === Clean moisture ===
    df_db["Mc_%"] = pd.to_numeric(df_db.get("Mc_%"), errors="coerce")

    # === Drop failed/anom rows if 'Coments' exists ===
    if "Coments" in df_db.columns:
        bad = df_db["Coments"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df_db = df_db[~bad]

    # keep only rows with a sample code and valid moisture
    df_db = df_db.dropna(subset=["Sample Code", "Mc_%"])

    # === Prepare PSD table ===
    for col in ["D10", "D50", "D90"]:
        df_psd[col] = pd.to_numeric(df_psd.get(col), errors="coerce")

    # === Merge DB rows with PSD by Sample Code (no aggregation) ===
    df = pd.merge(df_db, df_psd, on="Sample Code", how="inner")

    # require PSD and Mc_% present and positive D10/D50
    df = df.dropna(subset=["D10", "D50", "D90", "Mc_%"])
    df = df[(df["D10"] > 0) & (df["D50"] > 0)]

    # === Compute indices per-row ===
    df["EFI"] = df["D50"] / df["D10"]
    df["FSI"] = (df["D90"] - df["D10"]) / df["D50"]   # dimensionless span
    df["RS"]  = df["FSI"]

    # === Color by exact Sample Code ===
    df["Sample Label"] = df["Sample Code"].astype(str)
    labels_series = df["Sample Label"]

    # === Generic scatter helper with color groups ===
    def scatter_grouped(x, y, groups, all_labels, xlab, ylab, title):
        plt.figure(figsize=(8.0, 5.8))

        uniq = sorted(groups.unique())
        cmap = plt.cm.get_cmap("tab20", len(uniq))
        color_map = {g: cmap(i) for i, g in enumerate(uniq)}

        for g in uniq:
            m = groups == g
            plt.scatter(x[m], y[m], s=65, alpha=0.9, edgecolor="black",
                        linewidths=0.4, color=color_map[g], label=g)

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

    # === Plot 1: EFI vs Mc_% (all rows) ===
    scatter_grouped(
        df["EFI"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values, xlab="EFI = D50 / D10 (-)",
        ylab="Final Moisture (Mc %)",
        title="Final Moisture vs EFI"
    )

    # === Plot 2: FSI vs Mc_% (all rows) ===
    scatter_grouped(
        df["FSI"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values, xlab="PSD Span = (D90 - D10) / D50 (-)",
        ylab="Final Moisture (Mc %)",
        title="Final Moisture vs Span"
    )

    # === Plot 3: D10 vs Mc_% (all rows) ===
    scatter_grouped(
        df["D10"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values, xlab="D10 (\u03bcm)",
        ylab="Final Moisture (Mc %)",
        title="Final Moisture vs D10"
    )

    # === Plot 4: D50 vs Mc_% (all rows) ===
    scatter_grouped(
        df["D50"].values, df["Mc_%"].values, df["Sample Label"],
        labels_series.values, xlab="D50 (\u03bcm)",
        ylab="Final Moisture (Mc %)",
        title="Final Moisture vs D50"
    )

    return df
