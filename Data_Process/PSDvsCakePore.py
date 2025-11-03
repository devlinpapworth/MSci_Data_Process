import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_PSD_vs_CakePore(
    xlsx_path,
    sheet_db="DB",
    sheet_psd="PSD",
    d50_col="D50",
    include_samples=None,
    color_map=None,                # <-- NEW: pass a dict like {"Si_F":"#1f77b4", ...}
):
    # === Load sheets ===
    df_db  = pd.read_excel(xlsx_path, sheet_name=sheet_db,  engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # === Required columns ===
    for col in ["Sample Code", "Cake_por"]:
        if col not in df_db.columns:
            raise ValueError(f"'{col}' missing from sheet '{sheet_db}'.")
    for col in ["Sample Code", d50_col]:
        if col not in df_psd.columns:
            raise ValueError(f"'{col}' missing from sheet '{sheet_psd}'.")

    # === Clean + filter ===
    if "flag" in df_db.columns:
        bad = df_db["flag"].astype(str).str.lower().str.contains(r"\bfail\b|\banom\b", na=False)
        df_db = df_db[~bad]

    df_db["Cake_por"] = pd.to_numeric(df_db["Cake_por"], errors="coerce")
    df_psd[d50_col]   = pd.to_numeric(df_psd[d50_col],   errors="coerce")

    df_db  = df_db.dropna(subset=["Sample Code", "Cake_por"])
    df_psd = df_psd.dropna(subset=["Sample Code", d50_col])

    # === Merge DB + PSD on Sample Code ===
    df = pd.merge(df_db[["Sample Code", "Cake_por"]],
                  df_psd[["Sample Code", d50_col]],
                  on="Sample Code", how="inner")

    # === Manual include filter (optional) ===
    if include_samples is not None:
        df = df[df["Sample Code"].astype(str).isin([str(s) for s in include_samples])]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Cake_por", d50_col])
    df["Sample Label"] = df["Sample Code"].astype(str)

    # === Plot helper ===
    def scatter_grouped(x, y, groups, xlab, ylab, title, legend_title="Sample Code"):
        plt.figure(figsize=(8.0, 5.8))
        uniq = sorted(pd.Series(groups).unique())

        # Build a color lookup: use provided map if given, else tab20
        if color_map is not None:
            # default to grey when a label not present in color_map
            lookup = {g: color_map.get(g, "#808080") for g in uniq}
        else:
            cmap = plt.cm.get_cmap("tab20", len(uniq))
            lookup = {g: cmap(i) for i, g in enumerate(uniq)}

        for g in uniq:
            m = (groups == g)
            plt.scatter(
                np.asarray(x)[m], np.asarray(y)[m],
                s=65, alpha=0.9, edgecolor="black", linewidths=0.4,
                color=lookup[g], label=g
            )

        # Optional overall trendline
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

        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title=legend_title, fontsize=8, bbox_to_anchor=(1.02, 1),
                   loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    # === Plot Cake Porosity vs D50 ===
    scatter_grouped(
        df[d50_col].values,
        df["Cake_por"].values,
        df["Sample Label"],
        xlab=f"{d50_col} (\u03bcm)",
        ylab="Cake Porosity",
        title=f"Cake Porosity vs {d50_col}"
    )

    return df
