import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_moisture_vs_psd_indices(xlsx_path, sheet_db="DB", sheet_psd="PSD"):
    """
    Plot every valid row from DB (no aggregation) merged with PSD.
    Excludes rows where 'Coments' contains 'fail' or 'anom' (case-insensitive).
    Color-code points by Sample Code. Shows two plots: Mc% vs EFI and Mc% vs FSI.
    """

    # === Load data ===
    df_db = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # === Clean moisture ===
    # coerce 'Mc_%' to numeric; strings like '>25%' become NaN and are dropped
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
    df["RS"]  = (df["D90"] - df["D10"]) / df["D50"]   # relative span
    df["FSI"] = df["D10"] * df["RS"]

    # === Color by exact Sample Code ===
    df["Sample Label"] = df["Sample Code"].astype(str)
    labels_series = df["Sample Label"]

    # === Generic scatter helper with color groups ===
    def scatter_grouped(x, y, groups, all_labels, xlab, ylab, title):
        plt.figure(figsize=(8.0, 5.8))

        uniq = sorted(groups.unique())
        cmap = plt.cm.get_cmap("tab20", len(uniq))  # more colors if many samples
        color_map = {g: cmap(i) for i, g in enumerate(uniq)}

        # plot each sample code as a separate color group
        for g in uniq:
            m = groups == g
            plt.scatter(x[m], y[m], s=65, alpha=0.9, edgecolor="black",
                        linewidths=0.4, color=color_map[g], label=g)

        # overall trendline across all points
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            plt.plot(xs, p(xs), linestyle="--", color="black", label="Trend")
        except Exception:
            pass  # if not enough points for a fit

        # (Optional) annotate points — comment out if too busy
        # for xi, yi, sc in zip(x, y, all_labels):
        #     plt.annotate(str(sc), (xi, yi), textcoords="offset points",
        #                  xytext=(4, 4), fontsize=7)

        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(True, linestyle="--", alpha=0.5)
        # put legend outside if many groups
        plt.legend(title="Sample Code", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
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
        labels_series.values, xlab="PSD Span = D10 * (D90 - D10) / D50 (\u03bcm)",
        ylab="Final Moisture (Mc %)",
        title="Final Moisture vs Span"
    )

    return df
