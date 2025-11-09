# File: Data_Process/plot_hydraulic_vs_D10.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Iterable


def _safe_div(a, b):
    """Elementwise safe division that returns NaN when b==0 or either side is missing."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b
    out[(b == 0) | (b.isna())] = np.nan
    return out


def _compute_hydraulics(df: pd.DataFrame,
                        *,
                        mu_pa_s: float = 1.0e-3,   # water @ 
                        rho_kg_m3: float = 1000.0,
                        g_m_s2: float = 9.81,
                        f_prefix: str = "F_",
                        use_pressure_col: str = "F_P",
                        depth_col: str = "Depth",
                        area_col: str = "SA",
                        vol_col: str = "F_V",
                        time_col: str = "F_T") -> pd.DataFrame:
   

    df = df.copy()

    # --- Units / conversions ---
    dP_Pa = pd.to_numeric(df[use_pressure_col], errors="coerce") * 1.0e5   # a
    L_m   = pd.to_numeric(df[depth_col], errors="coerce") * 1.0e-3         # mm - m
    A_m2  = pd.to_numeric(df[area_col], errors="coerce")                   # m^2
    F_V_m3 = pd.to_numeric(df[vol_col],  errors="coerce") * 1.0e-6         # mL - m3
    F_T_s  = pd.to_numeric(df[time_col], errors="coerce")                  # s

    # Flow
    Q_m3s = _safe_div(F_V_m3, F_T_s)

    
    denom = (rho_kg_m3 * g_m_s2) * L_m
    i_head = _safe_div(dP_Pa, denom)

    
    num_K = Q_m3s * L_m * (rho_kg_m3 * g_m_s2)
    den_K = A_m2 * dP_Pa
    K_ms = _safe_div(num_K, den_K)

    
    num_k = Q_m3s * mu_pa_s * L_m
    den_k = A_m2 * dP_Pa
    k_m2 = _safe_div(num_k, den_k)

    df["Q_m3s"] = Q_m3s
    df["dP_Pa"] = dP_Pa
    df["L_m"] = L_m
    df["i_head"] = i_head
    df["K_ms"] = K_ms
    df["k_m2"] = k_m2
    return df


def plot_hydraulic_vs_D10(
    xlsx_path: str,
    sheet_db: str = "DB",
    sheet_psd: str = "PSD",
    *,
    include_samples: Optional[Iterable[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    use_pressure_col: str = "F_P",
    only_flag_include: bool = True,
    mu_pa_s: float = 1.0e-3,
    rho_kg_m3: float = 1000.0,
    g_m_s2: float = 9.81,
) -> pd.DataFrame:
    

    # --- Load sheets ---
    df_db  = pd.read_excel(xlsx_path, sheet_name=sheet_db,  engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # --- Basic checks ---
    need_db = ["Sample Code", "F_V", "F_T", use_pressure_col, "Depth", "SA"]
    for col in need_db:
        if col not in df_db.columns:
            raise ValueError(f"'{col}' missing from sheet '{sheet_db}'.")

    need_psd = ["Sample Code", "D10", "D50", "D90"]
    for col in need_psd:
        if col not in df_psd.columns:
            raise ValueError(f"'{col}' missing from sheet '{sheet_psd}'.")

    # --- Filter rows ---
    df_db = df_db.copy()
    if "flag" in df_db.columns and only_flag_include:
        mask = df_db["flag"].astype(str).str.lower().str.contains("include", na=False)
        df_db = df_db[mask]

    if include_samples is not None:
        df_db = df_db[df_db["Sample Code"].isin(list(include_samples))]

    # --- Compute hydraulics from filling step ---
    df_db = _compute_hydraulics(
        df_db,
        mu_pa_s=mu_pa_s,
        rho_kg_m3=rho_kg_m3,
        g_m_s2=g_m_s2,
        use_pressure_col=use_pressure_col,
        depth_col="Depth",
        area_col="SA",
        vol_col="F_V",
        time_col="F_T",
    )

    df_psd_key = (
        df_psd.dropna(subset=["Sample Code"])
              .groupby("Sample Code", as_index=False)[["D10", "D50", "D90"]]
              .median()
    )
    # --- Merge PSD (to get D10 etc. for x-axis) ---
    df_all = pd.merge(
        df_db,
        df_psd_key,
        on="Sample Code",
        how="left",
        validate="m:1",
    )

    # Keep rows that actually have D10 for plotting
    df_all = df_all[~pd.to_numeric(df_all["D10"], errors="coerce").isna()].copy()

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    ax1, ax2 = axes.ravel()

    # Helper to scatter with color by Sample Code
    def scatter_by_sample(ax, xcol, ycol, ylog=False, xlabel=None, ylabel=None):
        shown = set()
        for _, r in df_all.iterrows():
            sc = str(r["Sample Code"])
            c = None
            if color_map is not None and sc in color_map:
                c = color_map[sc]
            ax.scatter(r[xcol], r[ycol], s=55, edgecolor="k", linewidth=0.5, color=c, label=(sc if sc not in shown else None))
            shown.add(sc)
        if ylog:
            ax.set_yscale("log")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sample", frameon=True, fontsize=8)

    # 1) k vs D10 (log-y)
    scatter_by_sample(ax1, "D10", "k_m2", ylog=True, xlabel="D10 (\u03bcm)", ylabel="Permeability k (m^2, log)")

    # 2) K vs D10 (log-y)
    scatter_by_sample(ax2, "D10", "K_ms", ylog=True, xlabel="D10 (\u03bcm)", ylabel="Hydraulic conductivity K (m/s, log)")

    fig.suptitle("Hydraulic metrics vs D10 (computed from filling step)", fontsize=13)
    plt.show()

    return df_all


# ----------------------------
# Example usage from your main:
# ----------------------------
# from Data_Process.plot_hydraulic_vs_D10 import plot_hydraulic_vs_D10
#
# color_map = {
#     "Si_C": "#1f77b4",
#     "Si_M": "#ff7f0e",
#     "Si_F": "#2ca02c",
#     "Si_Rep": "#d62728",
#     "Si_Rep_new": "#9467bd",
#     "Si_BM": "#8c564b",
# }
#
# 
#
# print(df_out.filter(["Sample Code","D10","Q_m3s","i_head","K_ms","k_m2"]).round(6))
