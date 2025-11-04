# -*- coding: utf-8 -*-
import os
import warnings
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _require(df, cols, sheet):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in '{sheet}': {miss}\nAvailable: {list(df.columns)}")

def _maybe_to_m3(series: pd.Series) -> pd.Series:
    name = str(series.name).lower()
    s = pd.to_numeric(series, errors="coerce")
    if "ml" in name:
        return s * 1e-6
    if name.endswith("(l)") or name.endswith("[l]") or name.endswith("_l"):
        return s * 1e-3
    return s  # assume already m^3

def _maybe_to_pa(series: pd.Series) -> pd.Series:
    name = str(series.name).lower()
    s = pd.to_numeric(series, errors="coerce")
    if "kpa" in name:
        return s * 1_000.0
    return s  # assume Pa or bar-like (user should ensure units)

def _robust_Rm(w: np.ndarray, Rapp: np.ndarray) -> float:
    """Estimate Rm from small-w data: take 5th percentile of Rapp for w in lowest quartile."""
    ok = np.isfinite(w) & np.isfinite(Rapp) & (w > 0) & (Rapp > 0)
    if ok.sum() < 8:
        return 0.0  # fall back: upper-bound alpha (Rm=0)
    wq = np.quantile(w[ok], 0.25)
    mask = ok & (w <= wq)
    if mask.sum() < 5:
        return 0.0
    return float(np.quantile(Rapp[mask], 0.05))

def plot_cake_filtration_efficiency(
    xlsx_path: str,
    sheet_filter: str = "DB",   # keep signature; we use DB aggregates
    sheet_db: str = "DB",
    *,
    color_map: Optional[Dict[str, str]] = None,
    out_dir: str = "00_Figures",
    annotate: bool = True,
    mu_default: float = 1e-3   # Pa_s (water at ~20°C)
) -> pd.DataFrame:
    
    df = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")

    # Keep 'include' if flag present
    if "flag" in df.columns:
        df = df[df["flag"].astype(str).str.lower().str.contains("include", na=False)].copy()

    # Required columns
    _require(df, ["Sample Code", "P_T", "P_V", "P_P", "SA", "Mass_dry"], sheet_db)

    # Units & conversions
    # Time (s)
    t = pd.to_numeric(df["P_T"], errors="coerce").values
    # Volume -> m^3 (by name), else assume already m^3
    df["P_V_m3"] = _maybe_to_m3(df["P_V"])
    V = df["P_V_m3"].values
    # Pressure -> Pa (kPa by name), else assume Pa
    df["P_P_Pa"] = _maybe_to_pa(df["P_P"])
    dP = df["P_P_Pa"].values
    # Area (m^2)
    A = pd.to_numeric(df["SA"], errors="coerce").values
    # Dry mass (kg)
    mdry = pd.to_numeric(df["Mass_dry"], errors="coerce").values

    # Viscosity (Pa_s)
    mu_col = None
    for cand in ["mu_Pa_s", "\u03bc (Pa_s)", "\u03bcm"]:
        if cand in df.columns:
            mu_col = cand; break
    mu = pd.to_numeric(df[mu_col], errors="coerce").values if mu_col else np.full(len(df), mu_default, float)

    # Sanity & masks
    ok = np.isfinite(t) & (t > 0) & np.isfinite(V) & (V > 0) & np.isfinite(dP) & (dP > 0) \
         & np.isfinite(A) & (A > 0) & np.isfinite(mdry) & (mdry > 0) & np.isfinite(mu) & (mu > 0)
    if ok.sum() == 0:
        raise RuntimeError("No valid rows after sanity checks. Check units in P_T, P_V, P_P, SA, Mass_dry, viscosity.")

    dfg = df.loc[ok, ["Sample Code"]].copy()
    dfg["Q_m3_per_s"] = V[ok] / t[ok]
    dfg["A_m2"] = A[ok]
    dfg["w_kg_per_m2"] = mdry[ok] / A[ok]
    dfg["dP_Pa"] = dP[ok]
    dfg["mu_Pa_s"] = mu[ok]

    # Apparent total resistance Rapp = A*dP/(mu*Q)
    dfg["Rapp_per_m"] = (dfg["A_m2"] * dfg["dP_Pa"]) / (dfg["mu_Pa_s"] * dfg["Q_m3_per_s"])

    # Estimate Rm from low-w data; also compute upper-bound alpha* (Rm=0)
    Rm_est = _robust_Rm(dfg["w_kg_per_m2"].values, dfg["Rapp_per_m"].values)
    dfg["alpha_upperbound"] = dfg["Rapp_per_m"] / dfg["w_kg_per_m2"]
    dfg["alpha_m_per_kg"]   = (dfg["Rapp_per_m"] - Rm_est) / dfg["w_kg_per_m2"]
    dfg = dfg.replace([np.inf, -np.inf], np.nan).dropna(subset=["alpha_upperbound"])

    # Aggregate by Sample Code (median across runs)
    results = (dfg.groupby("Sample Code", as_index=False)
                    .agg(alpha_m_per_kg=("alpha_m_per_kg", "median"),
                         alpha_upperbound=("alpha_upperbound", "median"),
                         Rm_per_m=("Rapp_per_m", lambda x: Rm_est),  # same Rm applied
                         n_runs=("alpha_upperbound", "count")))

    # Plot efficiency (lower alpha = better)
    os.makedirs(out_dir, exist_ok=True)
    samples = results["Sample Code"].tolist()
    vals = results["alpha_m_per_kg"].to_numpy()

    # Fallback colours if needed
    if color_map is None:
        base = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c",
                                                                      "#d62728", "#9467bd", "#8c564b"])
        color_map = {s: base[i % len(base)] for i, s in enumerate(samples)}
    colors = [color_map.get(s, "#6e6e6e") for s in samples]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(samples))
    ax.scatter(x, vals, s=90, edgecolor="black", linewidth=0.6, zorder=3)
    for i, s in enumerate(samples):
        ax.scatter(i, vals[i], s=90, color=colors[i], edgecolor="black", linewidth=0.6, zorder=3)
        if annotate:
            ax.annotate(s, (i, vals[i]), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1e}"))
    ax.set_ylabel(r"Specific cake resistance, $\alpha$ (m kg$^{-1}$)  [median, R$_m$ from low-$w$]")
    ax.set_title("Cake Filtration Efficiency by Sample Code (pressing phase)")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "CF_efficiency_alpha_by_sample_DB.png"), dpi=220)
    plt.close(fig)

    # Save results
    results.to_csv(os.path.join(out_dir, "CF_results_by_sample_DB.csv"), index=False)

    return results
