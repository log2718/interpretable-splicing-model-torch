"""
Separate KL divergence figures (fig2 and fig3) for up_1_20 and ds_1_20.
Y-axis standardized across both regions.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT  = Path(__file__).resolve().parent / "rnaduplex_mfe_residual"
OUT.mkdir(parents=True, exist_ok=True)

ann = pd.read_csv(BASE / "data/test_annotated.csv")
fg  = pd.read_csv(BASE / "data/test_rnaduplex_finegrained.csv")

df = ann[["exon", "kl_flank_150_30_unc"]].merge(fg[["exon", "mfe_up_1_20", "mfe_ds_1_20"]], on="exon", how="inner")
df = df[df["kl_flank_150_30_unc"].notna()].copy()
print(f"{len(df):,} exons loaded")

REGIONS = ["up_1_20", "ds_1_20"]
N_BINS  = 20
COL_KL  = "#d6604d"


def clip(x, lo=0.1, hi=99.9):
    return np.clip(x, np.percentile(x, lo), np.percentile(x, hi))


def binned_kl(mfe, kl):
    edges = np.unique(np.percentile(mfe, np.linspace(0, 100, N_BINS + 1)))
    bin_idx, midpoints, means = [], [], []
    for i in range(len(edges) - 1):
        mask = (mfe >= edges[i]) & (mfe < edges[i + 1])
        if mask.sum() < 2:
            continue
        bin_idx.append(i + 1)
        midpoints.append((edges[i] + edges[i + 1]) / 2)
        means.append(kl[mask].mean())
    return np.array(bin_idx), np.array(midpoints), np.array(means)


def lowess_kl(mfe, kl, frac=0.3):
    mfe_c = clip(mfe)
    order = np.argsort(mfe_c)
    mfe_s = mfe_c[order]
    kl_s  = kl[order]
    sm    = lowess(kl_s, mfe_s, frac=frac, return_sorted=False)
    return mfe_s, sm


# ── Pre-pass: shared y-limits ─────────────────────────────────────────────────

all_bin_means, all_smooth = [], []
for region in REGIONS:
    mfe = df[f"mfe_{region}"].values
    kl  = df["kl_flank_150_30_unc"].values
    valid = np.isfinite(mfe) & np.isfinite(kl)
    _, _, means = binned_kl(mfe[valid], kl[valid])
    all_bin_means.extend(means.tolist())
    _, sm = lowess_kl(mfe[valid], kl[valid])
    all_smooth.extend(sm.tolist())

pad = 0.001
YLIM_FIG2 = (min(all_bin_means) - pad, max(all_bin_means) + pad)
YLIM_FIG3 = (min(all_smooth)    - pad, max(all_smooth)    + pad)
print(f"KL y-limits — fig2: {YLIM_FIG2[0]:.4f} to {YLIM_FIG2[1]:.4f}")
print(f"KL y-limits — fig3: {YLIM_FIG3[0]:.4f} to {YLIM_FIG3[1]:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────

for region in REGIONS:
    mfe = df[f"mfe_{region}"].values
    kl  = df["kl_flank_150_30_unc"].values
    valid = np.isfinite(mfe) & np.isfinite(kl)
    mfe, kl = mfe[valid], kl[valid]

    # Fig 2
    bin_idx, midpoints, means = binned_kl(mfe, kl)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bin_idx, means, color=COL_KL, linewidth=1.8, marker="o", markersize=5)
    for xi, mi, yi in zip(bin_idx, midpoints, means):
        ax.annotate(f"{mi:.1f}", xy=(xi, yi),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6, color="#333333")
    ax.set_xticks(bin_idx)
    ax.set_ylim(YLIM_FIG2)
    ax.set_xlabel("MFE bin (midpoint kcal/mol labeled)", fontsize=10)
    ax.set_ylabel("Mean KL divergence", fontsize=10)
    ax.set_title(f"Binned Mean KL divergence vs RNAduplex MFE — {region}", fontsize=11)
    fig.tight_layout()
    p = OUT / f"{region}_fig2_kl.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p.name}")

    # Fig 3
    mfe_s, sm = lowess_kl(mfe, kl)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mfe_s, sm, color=COL_KL, linewidth=1.8)
    ax.set_ylim(YLIM_FIG3)
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("KL divergence", fontsize=10)
    ax.set_title(f"LOWESS KL divergence vs RNAduplex MFE — {region}", fontsize=11)
    fig.tight_layout()
    p = OUT / f"{region}_fig3_kl.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p.name}")

print("\nDone.")
