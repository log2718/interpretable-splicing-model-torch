"""
Residual PSI vs RNAduplex MFE — PSI-stratified LOWESS overlay.

Regions: up_1_20, ds_1_20
Filter:  |residual| > 0.1 (remove middle band)
Strata:  10 PSI bins of width 0.1  (0-0.1, 0.1-0.2, ..., 0.9-1.0)

One figure per region: overlaid LOWESS curves, one per stratum.
Controls for the extremity bias where PSI≈0 / PSI≈1 limit the sign of residuals.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT  = Path(__file__).resolve().parent / "original_model_finegrained"
OUT.mkdir(parents=True, exist_ok=True)

REGIONS   = ["up_1_20", "ds_1_20"]
PSI_EDGES = np.arange(0, 1.2, 0.2)
FRAC      = 0.2
MIN_PTS   = 30

# ── Load ──────────────────────────────────────────────────────────────────────

ann = pd.read_csv(BASE / "data/test_annotated.csv")
fg  = pd.read_csv(BASE / "data/test_rnaduplex_finegrained.csv")

mfe_cols = [f"mfe_{r}" for r in REGIONS]
df = ann[["exon", "PSI", "predicted_PSI_baseline"]].merge(
    fg[["exon"] + mfe_cols], on="exon", how="inner"
)
df = df[df["PSI"].notna() & df["predicted_PSI_baseline"].notna()].copy()
df["residual"] = df["predicted_PSI_baseline"] - df["PSI"]

# Apply |residual| > 0.1 filter
filt = df[df["residual"].abs() > 0.1].copy()
print(f"After filter: {len(filt):,} exons")

# ── Colour map: 10 bins → diverging (blue=low PSI, red=high PSI) ──────────────

cmap    = matplotlib.colormaps.get_cmap("RdYlBu_r").resampled(len(PSI_EDGES) - 1)
colours = [cmap(i) for i in range(len(PSI_EDGES) - 1)]
colours[2] = "#b8860b"  # darker yellow for PSI 0.4–0.6

# ── Per-region figure ─────────────────────────────────────────────────────────

for region in REGIONS:
    mfe_col = f"mfe_{region}"
    sub = filt[filt[mfe_col].notna()].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = []
    for i in range(len(PSI_EDGES) - 1):
        lo_psi, hi_psi = PSI_EDGES[i], PSI_EDGES[i + 1]
        lo_op = (sub["PSI"] >= lo_psi) if i == 0 else (sub["PSI"] > lo_psi)
        mask = lo_op & (sub["PSI"] <= hi_psi)
        chunk = sub[mask]
        if len(chunk) < MIN_PTS:
            continue

        mfe_s = chunk[mfe_col].values
        res_s = chunk["residual"].values
        order = np.argsort(mfe_s)
        sm    = lowess(res_s[order], mfe_s[order], frac=FRAC, it=0, return_sorted=False)

        label = f"PSI {lo_psi:.1f}–{hi_psi:.1f}  (n={len(chunk):,})"
        ax.plot(mfe_s[order], sm, color=colours[i], linewidth=1.6,
                label=label, alpha=0.85)
        plotted.append(i)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI  (predicted − actual)", fontsize=10)
    ax.set_title(
        f"LOWESS residual PSI vs MFE — {region}\n"
        f"(|residual|>0.1 filter; stratified by actual PSI)",
        fontsize=11
    )
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85,
              title="PSI stratum", title_fontsize=8)
    fig.tight_layout()
    out_path = OUT / f"{region}_psi_stratified_lowess.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.name}  ({len(plotted)} strata plotted)")

# ── Full dataset version (no excl filter) ────────────────────────────────────

print(f"\nFull dataset: {len(df):,} exons")

for region in REGIONS:
    mfe_col = f"mfe_{region}"
    sub = df[df[mfe_col].notna()].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = []
    for i in range(len(PSI_EDGES) - 1):
        lo_psi, hi_psi = PSI_EDGES[i], PSI_EDGES[i + 1]
        lo_op = (sub["PSI"] >= lo_psi) if i == 0 else (sub["PSI"] > lo_psi)
        mask = lo_op & (sub["PSI"] <= hi_psi)
        chunk = sub[mask]
        if len(chunk) < MIN_PTS:
            continue

        mfe_s = chunk[mfe_col].values
        res_s = chunk["residual"].values
        order = np.argsort(mfe_s)
        sm    = lowess(res_s[order], mfe_s[order], frac=FRAC, it=0, return_sorted=False)

        label = f"PSI {lo_psi:.1f}–{hi_psi:.1f}  (n={len(chunk):,})"
        ax.plot(mfe_s[order], sm, color=colours[i], linewidth=1.6,
                label=label, alpha=0.85)
        plotted.append(i)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=10)
    ax.set_ylabel("Residual PSI  (predicted − actual)", fontsize=10)
    ax.set_title(
        f"LOWESS residual PSI vs MFE — {region}\n"
        f"(full dataset; stratified by actual PSI)",
        fontsize=11
    )
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85,
              title="PSI stratum", title_fontsize=8)
    fig.tight_layout()
    out_path = OUT / f"{region}_psi_stratified_full_lowess.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.name}  ({len(plotted)} strata plotted)")

print("Done.")
