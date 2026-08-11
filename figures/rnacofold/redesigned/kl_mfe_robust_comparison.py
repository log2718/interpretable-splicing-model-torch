"""
KL divergence vs RNAduplex MFE — up_1_20 and ds_1_20.

Two plots per region:
  A) LOWESS with it=0 (no robust reweighting) — full dataset
  B) 20 equal-frequency bins of MFE, showing true arithmetic mean KL per bin
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT  = Path(__file__).resolve().parent / "original_model_finegrained"
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = ["up_1_20", "ds_1_20"]
FRAC    = 0.2
N_BINS  = 20
COL_KL  = "#d6604d"

# ── Load ──────────────────────────────────────────────────────────────────────

ann = pd.read_csv(BASE / "data/test_annotated.csv")
fg  = pd.read_csv(BASE / "data/test_rnaduplex_finegrained.csv")

mfe_cols = [f"mfe_{r}" for r in REGIONS]
df = ann[["exon", "PSI", "predicted_PSI_baseline", "kl_baseline"]].merge(
    fg[["exon"] + mfe_cols], on="exon", how="inner"
)
df = df[df["PSI"].notna() & df["predicted_PSI_baseline"].notna()].copy()
df["residual"] = df["predicted_PSI_baseline"] - df["PSI"]
print(f"Total exons: {len(df):,}")



def get_binned_equal_freq(mfe, y, n=N_BINS):
    edges = np.unique(np.percentile(mfe, np.linspace(0, 100, n + 1)))
    midpoints, means, ns = [], [], []
    for i in range(len(edges) - 1):
        mask = (mfe >= edges[i]) & (mfe < edges[i + 1])
        if mask.sum() < 2:
            continue
        midpoints.append((edges[i] + edges[i + 1]) / 2)
        means.append(y[mask].mean())
        ns.append(mask.sum())
    return np.array(midpoints), np.array(means), np.array(ns)


filt = df[df["residual"].abs() > 0.1].copy()
print(f"After |residual|>0.1 filter: {len(filt):,}")

for region in REGIONS:
    mfe_col = f"mfe_{region}"
    sub = df[df[mfe_col].notna() & df["kl_baseline"].notna()].copy()
    mfe = sub[mfe_col].values
    kl  = sub["kl_baseline"].values

    order = np.argsort(mfe)

    # ── Plot A: LOWESS it=0 ───────────────────────────────────────────────────

    sm_it0 = lowess(kl[order], mfe[order], frac=FRAC, it=0, return_sorted=False)
    sm_it3 = lowess(kl[order], mfe[order], frac=FRAC, it=3, return_sorted=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mfe[order], sm_it3, color="#888888", linewidth=1.6,
            linestyle="--", label="LOWESS it=3 (robust)")
    ax.plot(mfe[order], sm_it0, color=COL_KL, linewidth=1.8,
            label="LOWESS it=0 (no robust reweighting)")
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("KL divergence", fontsize=10)
    ax.set_title(f"KL divergence vs MFE — {region}\n"
                 f"LOWESS it=0 vs it=3 (full dataset, n={len(sub):,})", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_a = OUT / f"{region}_kl_lowess_it0.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_a.name}")

    # ── Plot B: binned mean KL (equal-frequency) ──────────────────────────────

    midpoints, means, ns = get_binned_equal_freq(mfe, kl)
    bin_idx = np.arange(1, len(midpoints) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_idx, means, color=COL_KL, linewidth=1.8, marker="o", markersize=5)
    for xi, mi, yi in zip(bin_idx, midpoints, means):
        ax.annotate(f"{mi:.1f}", xy=(xi, yi), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6, color="#333333")
    ax.set_xticks(bin_idx)
    ax.set_xlabel("MFE bin (midpoint kcal/mol labeled)", fontsize=10)
    ax.set_ylabel("Mean KL divergence", fontsize=10)
    ax.set_title(f"Binned mean KL divergence vs MFE — {region}\n"
                 f"(20 equal-frequency bins, full dataset, n={len(sub):,})", fontsize=11)
    fig.tight_layout()
    out_b = OUT / f"{region}_kl_binned.png"
    fig.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_b.name}")

    # ── Plot C: LOWESS it=0, excl |residual|≤0.1 ─────────────────────────────

    sub_f = filt[filt[mfe_col].notna() & filt["kl_baseline"].notna()].copy()
    mfe_f = sub_f[mfe_col].values
    kl_f  = sub_f["kl_baseline"].values
    order_f = np.argsort(mfe_f)

    sm_f_it0 = lowess(kl_f[order_f], mfe_f[order_f], frac=FRAC, it=0, return_sorted=False)
    sm_f_it3 = lowess(kl_f[order_f], mfe_f[order_f], frac=FRAC, it=3, return_sorted=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mfe_f[order_f], sm_f_it3, color="#888888", linewidth=1.6,
            linestyle="--", label="LOWESS it=3 (robust)")
    ax.plot(mfe_f[order_f], sm_f_it0, color=COL_KL, linewidth=1.8,
            label="LOWESS it=0 (no robust reweighting)")
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("KL divergence", fontsize=10)
    ax.set_title(f"KL divergence vs MFE — {region}\n"
                 f"LOWESS it=0 vs it=3 (|residual|>0.1, n={len(sub_f):,})", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_c = OUT / f"{region}_excl01_kl_lowess_it0.png"
    fig.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_c.name}")

print("Done.")
