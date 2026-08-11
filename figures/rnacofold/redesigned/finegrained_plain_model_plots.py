"""
Four figures per finegrained region using the plain flank_150_30 model
(KL-divergence trained, no uncertainty head).

Regions: all 8 from test_rnaduplex_finegrained.csv
  up_1_20, up_21_40, up_41_60
  ds_1_20, ds_21_40, ds_41_60, ds_61_80, ds_81_100

Fig 1: 2D histogram (RNAduplex MFE vs residual PSI)
Fig 2: Binned mean residual (20 equal-frequency bins, midpoint labeled)
Fig 3: LOWESS trendline — residual PSI
Fig 4: LOWESS trendline — KL divergence

Y-axes are shared across all 8 regions within each figure type.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT  = Path(__file__).resolve().parent / "original_model_finegrained"
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = [
    "up_1_20", "up_21_40", "up_41_60",
    "ds_1_20", "ds_21_40", "ds_41_60", "ds_61_80", "ds_81_100",
]
N_BINS  = 20
COL_RES = "#2166ac"
COL_KL  = "#d6604d"



# ── Load data ─────────────────────────────────────────────────────────────────

ann = pd.read_csv(BASE / "data/test_annotated.csv")
fg  = pd.read_csv(BASE / "data/test_rnaduplex_finegrained.csv")

mfe_cols = [f"mfe_{r}" for r in REGIONS]
df = ann[["exon", "PSI", "predicted_PSI_baseline", "kl_baseline"]].merge(
    fg[["exon"] + mfe_cols], on="exon", how="inner"
)
df = df[df["PSI"].notna() & df["predicted_PSI_baseline"].notna()].copy()
df["residual"] = df["predicted_PSI_baseline"] - df["PSI"]
print(f"{len(df):,} exons loaded")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_binned(mfe, y, n=N_BINS):
    edges = np.unique(np.percentile(mfe, np.linspace(0, 100, n + 1)))
    midpoints, means = [], []
    for i in range(len(edges) - 1):
        mask = (mfe >= edges[i]) & (mfe < edges[i + 1])
        if mask.sum() < 2:
            continue
        midpoints.append((edges[i] + edges[i + 1]) / 2)
        means.append(y[mask].mean())
    return np.array(midpoints), np.array(means)


def get_lowess(mfe, y, frac=0.2, it=3):
    order = np.argsort(mfe)
    sm = lowess(y[order], mfe[order], frac=frac, it=it, return_sorted=False)
    return mfe[order], sm


# ── Pre-pass: shared y-limits ─────────────────────────────────────────────────

all_res_binned, all_res_smooth = [], []
all_kl_binned,  all_kl_smooth  = [], []

for region in REGIONS:
    mfe = df[f"mfe_{region}"].values
    res = df["residual"].values
    kl  = df["kl_baseline"].values
    valid = np.isfinite(mfe) & np.isfinite(res) & np.isfinite(kl)
    mfe, res, kl = mfe[valid], res[valid], kl[valid]

    _, means = get_binned(mfe, res)
    all_res_binned.extend(means.tolist())
    _, sm = get_lowess(mfe, res, it=0)
    all_res_smooth.extend(sm.tolist())

    _, means_kl = get_binned(mfe, kl)
    all_kl_binned.extend(means_kl.tolist())
    _, sm_kl = get_lowess(mfe, kl, it=0)
    all_kl_smooth.extend(sm_kl.tolist())

pad_res = 0.005
pad_kl  = 0.001
YLIM_RES_BIN = (min(all_res_binned) - pad_res, max(all_res_binned) + pad_res)
YLIM_RES_LWS = (min(all_res_smooth) - pad_res, max(all_res_smooth) + pad_res)
YLIM_KL_BIN  = (min(all_kl_binned)  - pad_kl,  max(all_kl_binned)  + pad_kl)
YLIM_KL_LWS  = (min(all_kl_smooth)  - pad_kl,  max(all_kl_smooth)  + pad_kl)
print(f"Residual y-limits — binned: {YLIM_RES_BIN[0]:.4f} to {YLIM_RES_BIN[1]:.4f}")
print(f"Residual y-limits — lowess: {YLIM_RES_LWS[0]:.4f} to {YLIM_RES_LWS[1]:.4f}")
print(f"KL y-limits       — binned: {YLIM_KL_BIN[0]:.4f} to {YLIM_KL_BIN[1]:.4f}")
print(f"KL y-limits       — lowess: {YLIM_KL_LWS[0]:.4f} to {YLIM_KL_LWS[1]:.4f}")

# ── Per-region plots ──────────────────────────────────────────────────────────

for region in REGIONS:
    mfe = df[f"mfe_{region}"].values
    res = df["residual"].values
    kl  = df["kl_baseline"].values
    valid = np.isfinite(mfe) & np.isfinite(res) & np.isfinite(kl)
    mfe, res, kl = mfe[valid], res[valid], kl[valid]
    print(f"\n{region}  (n={len(mfe):,})")

    # Fig 1: 2D histogram
    r, _ = pearsonr(mfe, res)
    H, xe, ye = np.histogram2d(mfe, res,
                               bins=[np.linspace(mfe.min(), mfe.max(), 80),
                                     np.linspace(res.min(), res.max(), 80)])
    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(xe, ye, H.T, cmap="Blues",
                        norm=mcolors.LogNorm(vmin=0.6, vmax=H.max()))
    fig.colorbar(pcm, ax=ax, label="count")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.text(0.02, 0.97, f"n={len(mfe):,}\nr={r:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI", fontsize=10)
    ax.set_title(f"RNAduplex MFE vs Residual PSI — {region}\n(original PNAS model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_fig1_2dhist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: binned mean residual
    midpoints, means = get_binned(mfe, res)
    bin_idx = np.arange(1, len(midpoints) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_idx, means, color=COL_RES, linewidth=1.8, marker="o", markersize=5)
    for xi, mi, yi in zip(bin_idx, midpoints, means):
        ax.annotate(f"{mi:.1f}", xy=(xi, yi), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6, color="#333333")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_ylim(YLIM_RES_BIN)
    ax.set_xticks(bin_idx)
    ax.set_xlabel("MFE bin (midpoint kcal/mol labeled)", fontsize=10)
    ax.set_ylabel("Mean residual PSI", fontsize=10)
    ax.set_title(f"Binned Mean Residual vs RNAduplex MFE — {region}\n(original PNAS model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_fig2_binned.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: LOWESS residual (it=0 solid, it=3 grey dotted)
    mfe_s, sm_it0 = get_lowess(mfe, res, it=0)
    _,     sm_it3 = get_lowess(mfe, res, it=3)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mfe_s, sm_it3, color="#888888", linewidth=1.4, linestyle=":", label="LOWESS it=3 (robust)")
    ax.plot(mfe_s, sm_it0, color=COL_RES,  linewidth=1.8, label="LOWESS it=0")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_ylim(YLIM_RES_LWS)
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI", fontsize=10)
    ax.set_title(f"LOWESS: Residual PSI vs RNAduplex MFE — {region}\n(original PNAS model)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_fig3_lowess.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: LOWESS KL divergence (it=0 solid, it=3 grey dotted)
    mfe_s, sm_kl_it0 = get_lowess(mfe, kl, it=0)
    _,     sm_kl_it3 = get_lowess(mfe, kl, it=3)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mfe_s, sm_kl_it3, color="#888888", linewidth=1.4, linestyle=":", label="LOWESS it=3 (robust)")
    ax.plot(mfe_s, sm_kl_it0, color=COL_KL,   linewidth=1.8, label="LOWESS it=0")
    ax.set_ylim(YLIM_KL_LWS)
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("KL divergence", fontsize=10)
    ax.set_title(f"LOWESS KL divergence vs RNAduplex MFE — {region}\n(original PNAS model)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_fig4_kl_lowess.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Overlay: 2D histogram + LOWESS residual PSI (it=0 orange, it=3 grey dotted)
    if region in ("up_1_20", "ds_1_20"):
        r_ov, _ = pearsonr(mfe, res)
        H_ov, xe_ov, ye_ov = np.histogram2d(
            mfe, res,
            bins=[np.linspace(mfe.min(), mfe.max(), 80),
                  np.linspace(res.min(), res.max(), 80)]
        )
        mfe_ov, sm_ov_it0 = get_lowess(mfe, res, it=0)
        _,      sm_ov_it3 = get_lowess(mfe, res, it=3)
        fig, ax = plt.subplots(figsize=(10, 8))
        pcm = ax.pcolormesh(xe_ov, ye_ov, H_ov.T, cmap="Blues",
                            norm=mcolors.LogNorm(vmin=0.6, vmax=H_ov.max()))
        fig.colorbar(pcm, ax=ax, label="count")
        ax.plot(mfe_ov, sm_ov_it3, color="#888888", linewidth=1.4, linestyle=":",
                label="LOWESS it=3 (robust)")
        ax.plot(mfe_ov, sm_ov_it0, color="orange", linewidth=2.0,
                label="LOWESS residual PSI")
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.text(0.02, 0.97, f"n={len(mfe):,}\nr={r_ov:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
        ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
        ax.set_ylabel("Residual PSI", fontsize=10)
        ax.set_title(f"RNAduplex MFE vs Residual PSI — {region}\n(original PNAS model)", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / f"{region}_overlay_hist_lowess.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved figures for {region}")

# ── excl01 plots for up_1_20 and ds_1_20 only ────────────────────────────────

df_filt = df[df["residual"].abs() > 0.1].copy()
print(f"\nexcl01 subset: {len(df_filt):,} exons")

for region in ("up_1_20", "ds_1_20"):
    mfe = df_filt[f"mfe_{region}"].values
    res = df_filt["residual"].values
    kl  = df_filt["kl_baseline"].values
    valid = np.isfinite(mfe) & np.isfinite(res) & np.isfinite(kl)
    mfe, res, kl = mfe[valid], res[valid], kl[valid]
    print(f"\n{region} excl01  (n={len(mfe):,})")

    # excl01 fig1: 2D histogram
    r, _ = pearsonr(mfe, res)
    H, xe, ye = np.histogram2d(mfe, res,
                               bins=[np.linspace(mfe.min(), mfe.max(), 80),
                                     np.linspace(res.min(), res.max(), 80)])
    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(xe, ye, H.T, cmap="Blues",
                        norm=mcolors.LogNorm(vmin=0.6, vmax=H.max()))
    fig.colorbar(pcm, ax=ax, label="count")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.text(0.02, 0.97, f"n={len(mfe):,}\nr={r:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI", fontsize=10)
    ax.set_title(f"RNAduplex MFE vs Residual PSI — {region}\n(original PNAS model, |residual|>0.1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_excl01_fig1_2dhist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # excl01 fig2: binned mean residual
    midpoints, means = get_binned(mfe, res)
    bin_idx = np.arange(1, len(midpoints) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_idx, means, color=COL_RES, linewidth=1.8, marker="o", markersize=5)
    for xi, mi, yi in zip(bin_idx, midpoints, means):
        ax.annotate(f"{mi:.1f}", xy=(xi, yi), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6, color="#333333")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xticks(bin_idx)
    ax.set_xlabel("MFE bin (midpoint kcal/mol labeled)", fontsize=10)
    ax.set_ylabel("Mean residual PSI", fontsize=10)
    ax.set_title(f"Binned Mean Residual vs RNAduplex MFE — {region}\n(original PNAS model, |residual|>0.1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_excl01_fig2_binned.png", dpi=150, bbox_inches="tight")
    plt.close()

    # excl01 fig3: LOWESS residual PSI — it=0 only
    mfe_s, sm_it0 = get_lowess(mfe, res, it=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mfe_s, sm_it0, color=COL_RES, linewidth=1.8, label="LOWESS it=0")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI", fontsize=10)
    ax.set_title(f"LOWESS: Residual PSI vs RNAduplex MFE — {region}\n(original PNAS model, |residual|>0.1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_excl01_fig3_lowess.png", dpi=150, bbox_inches="tight")
    plt.close()

    # excl01 fig4: LOWESS KL — it=0 only
    mfe_s, sm_kl = get_lowess(mfe, kl, it=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mfe_s, sm_kl, color=COL_KL, linewidth=1.8, label="LOWESS it=0")
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("KL divergence", fontsize=10)
    ax.set_title(f"LOWESS KL divergence vs RNAduplex MFE — {region}\n(original PNAS model, |residual|>0.1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_excl01_fig4_kl_lowess.png", dpi=150, bbox_inches="tight")
    plt.close()

    # excl01 overlay: 2D histogram + LOWESS residual PSI it=0 only
    fig, ax = plt.subplots(figsize=(10, 8))
    pcm = ax.pcolormesh(xe, ye, H.T, cmap="Blues",
                        norm=mcolors.LogNorm(vmin=0.6, vmax=H.max()))
    fig.colorbar(pcm, ax=ax, label="count")
    ax.plot(mfe_s, sm_it0, color="orange", linewidth=2.0, label="LOWESS residual PSI")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.text(0.02, 0.97, f"n={len(mfe):,}\nr={r:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI", fontsize=10)
    ax.set_title(f"RNAduplex MFE vs Residual PSI — {region}\n(original PNAS model, |residual|>0.1 only)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"{region}_excl01_overlay_hist_lowess.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved excl01 figures for {region}")

print(f"\nAll done. Output: {OUT}")
