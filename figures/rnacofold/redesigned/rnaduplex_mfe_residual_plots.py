"""
Three figures per region showing RNAduplex MFE vs residual PSI.

Regions:
  Coarse: ss3, up_near, up_mid, up_far, up_total, upstream_exon,
          downstream_short, downstream_long, downstream_exon
  Fine-grained (significant): up_1_20, ds_1_20

Fig 1: 2D histogram (pcolormesh, LogNorm, Blues)
Fig 2: Binned mean residual (20 equal-frequency bins, 95% CI shaded)
Fig 3: LOWESS trendline with 100-resample bootstrap CI
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
OUT  = Path(__file__).resolve().parent / "rnaduplex_mfe_residual"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load and merge ─────────────────────────────────────────────────────────────

ann = pd.read_csv(BASE / "data/test_annotated.csv")
rd  = pd.read_csv(BASE / "data/test_rnaduplex_redesigned.csv")
fg  = pd.read_csv(BASE / "data/test_rnaduplex_finegrained.csv")

df = ann[["exon", "PSI", "predicted_PSI_flank_150_30_unc"]].merge(rd, on="exon", how="inner")
df = df.merge(fg[["exon", "mfe_up_1_20", "mfe_ds_1_20"]], on="exon", how="inner")
df = df[df["PSI"].notna() & df["predicted_PSI_flank_150_30_unc"].notna()].copy()
df["residual"] = df["predicted_PSI_flank_150_30_unc"] - df["PSI"]
print(f"{len(df):,} exons loaded")

COARSE_REGIONS = [
    "ss3", "up_near", "up_mid", "up_far", "up_total",
    "upstream_exon", "downstream_short", "downstream_long", "downstream_exon",
]
FINE_REGIONS = ["up_1_20", "ds_1_20"]
ALL_REGIONS  = COARSE_REGIONS + FINE_REGIONS


def clip(x, lo=0.1, hi=99.9):
    return np.clip(x, np.percentile(x, lo), np.percentile(x, hi))


YLIM_FIG2 = None  # set after first pass over all regions
YLIM_FIG3 = None


# ── Fig 1: 2D histogram ───────────────────────────────────────────────────────

def fig1_2dhist(mfe, res, region):
    mfe_c = clip(mfe)
    res_c = clip(res)
    r, _ = pearsonr(mfe_c, res_c)

    bins_x = np.linspace(mfe_c.min(), mfe_c.max(), 80)
    bins_y = np.linspace(res_c.min(), res_c.max(), 80)
    H, xedge, yedge = np.histogram2d(mfe_c, res_c, bins=[bins_x, bins_y])

    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(
        xedge, yedge, H.T,
        cmap="Blues",
        norm=mcolors.LogNorm(vmin=0.6, vmax=H.max()),
    )
    fig.colorbar(pcm, ax=ax, label="count")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.text(0.02, 0.97, f"n={len(mfe_c):,}\nr={r:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI", fontsize=10)
    ax.set_title(f"RNAduplex MFE vs Residual PSI — {region}", fontsize=11)
    fig.tight_layout()
    p = OUT / f"{region}_fig1_2dhist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p.name}")


# ── Fig 2: Binned mean residual ───────────────────────────────────────────────

def fig2_binned(mfe, res, region, n_bins=20):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(mfe, percentiles)
    # Deduplicate edges (can happen in sparse regions)
    edges = np.unique(edges)

    midpoints, means, cis = [], [], []
    for i in range(len(edges) - 1):
        mask = (mfe >= edges[i]) & (mfe < edges[i + 1])
        if mask.sum() < 2:
            continue
        vals = res[mask]
        midpoints.append((edges[i] + edges[i + 1]) / 2)
        means.append(vals.mean())
        cis.append(1.96 * vals.std() / np.sqrt(len(vals)))

    midpoints = np.array(midpoints)
    means     = np.array(means)
    cis       = np.array(cis)

    bin_idx = np.arange(1, len(midpoints) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_idx, means, color="#2166ac", linewidth=1.8, marker="o", markersize=5)
    for xi, mi, yi in zip(bin_idx, midpoints, means):
        ax.annotate(f"{mi:.1f}", xy=(xi, yi),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6, color="#333333")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    if YLIM_FIG2 is not None:
        ax.set_ylim(YLIM_FIG2)
    ax.set_xticks(bin_idx)
    ax.set_xlabel("MFE bin (midpoint kcal/mol labeled)", fontsize=10)
    ax.set_ylabel("Mean residual PSI", fontsize=10)
    ax.set_title(f"Binned Mean Residual vs RNAduplex MFE — {region}", fontsize=11)
    fig.tight_layout()
    p = OUT / f"{region}_fig2_binned.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p.name}")


# ── Fig 3: LOWESS + bootstrap CI ─────────────────────────────────────────────

def fig3_lowess(mfe, res, region, frac=0.3):
    mfe_c = clip(mfe)
    res_c = res[np.argsort(mfe_c)]
    mfe_s = np.sort(mfe_c)

    smooth = lowess(res_c, mfe_s, frac=frac, return_sorted=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mfe_s, smooth, color="#2166ac", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    if YLIM_FIG3 is not None:
        ax.set_ylim(YLIM_FIG3)
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI", fontsize=10)
    ax.set_title(f"LOWESS: Residual PSI vs RNAduplex MFE — {region}", fontsize=11)
    fig.tight_layout()
    p = OUT / f"{region}_fig3_lowess.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p.name}")


# ── Pre-pass: compute shared y-limits across all regions ─────────────────────

all_means, all_smooth = [], []
for region in ALL_REGIONS:
    col = f"mfe_{region}"
    mfe = df[col].values
    res = df["residual"].values
    valid = np.isfinite(mfe) & np.isfinite(res)
    mfe, res = mfe[valid], res[valid]

    # Fig2 binned means
    edges = np.unique(np.percentile(mfe, np.linspace(0, 100, 21)))
    for i in range(len(edges) - 1):
        mask = (mfe >= edges[i]) & (mfe < edges[i + 1])
        if mask.sum() >= 2:
            all_means.append(res[mask].mean())

    # Fig3 LOWESS
    mfe_c = clip(mfe)
    res_s = res[np.argsort(mfe_c)]
    sm    = lowess(res_s, np.sort(mfe_c), frac=0.3, return_sorted=False)
    all_smooth.extend(sm.tolist())

pad = 0.005
YLIM_FIG2 = (min(all_means) - pad, max(all_means) + pad)
YLIM_FIG3 = (min(all_smooth) - pad, max(all_smooth) + pad)
print(f"Shared y-limits — fig2: {YLIM_FIG2[0]:.4f} to {YLIM_FIG2[1]:.4f}")
print(f"Shared y-limits — fig3: {YLIM_FIG3[0]:.4f} to {YLIM_FIG3[1]:.4f}")

# ── Run all regions ───────────────────────────────────────────────────────────

for region in ALL_REGIONS:
    col = f"mfe_{region}"
    print(f"\n{region}")
    mfe = df[col].values
    res = df["residual"].values
    valid = np.isfinite(mfe) & np.isfinite(res)
    mfe, res = mfe[valid], res[valid]
    fig1_2dhist(mfe, res, region)
    fig2_binned(mfe, res, region)
    fig3_lowess(mfe, res, region)

print("\nDone.")
