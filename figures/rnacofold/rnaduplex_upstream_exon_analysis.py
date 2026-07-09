"""
Exon-exon interaction analysis: fixed upstream exon vs variable exon.

Tests whether the variable exon (different per row) can form a stable
RNA duplex with the fixed upstream exon (constant across all MPRA experiments).
Bottom 2% MFE vs rest for mean residual PSI and variance of residual PSI.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene as levene_test

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

ann    = pd.read_csv(BASE / "data/test_annotated.csv")
duplex = pd.read_csv(BASE / "data/test_rnaduplex_upstream_exon.csv")
df = ann.merge(duplex, on="exon", how="inner")
df = df[
    df["PSI"].notna() &
    df["predicted_PSI_flank_150_30_unc"].notna() &
    df["var_flank_150_30_unc"].notna()
].reset_index(drop=True)

df["signed_residual"] = df["predicted_PSI_flank_150_30_unc"] - df["PSI"]
print(f"{len(df):,} exons with complete data")

COLORS = ["#92c5de", "#d6604d"]
rng    = np.random.default_rng(42)

col = "mfe_upstream_exon"
thr  = df[col].quantile(0.02)
bot  = df[df[col] <= thr]
rest = df[df[col] >  thr]

_, p_signed = ttest_ind(bot["signed_residual"].values, rest["signed_residual"].values)
_, p_levene = levene_test(bot["signed_residual"].values, rest["signed_residual"].values)

print(f"\nthreshold={thr:.1f} kcal/mol   n_bot={len(bot)}")
print(f"mean_bot={bot['signed_residual'].mean():+.4f}  "
      f"mean_rest={rest['signed_residual'].mean():+.4f}  "
      f"p_mean={p_signed:.2e}  p_levene={p_levene:.2e}")

def jitter_boxplot(ax, groups, tick_labels, ylabel, title):
    bp = ax.boxplot(groups, tick_labels=tick_labels, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5), widths=0.45)
    for box, color in zip(bp["boxes"], COLORS):
        box.set_facecolor(color); box.set_alpha(0.6)
    for j, (vals, color) in enumerate(zip(groups, COLORS)):
        alpha = min(0.4, 200 / len(vals))
        x = rng.uniform(j + 0.75, j + 1.25, size=len(vals))
        ax.scatter(x, vals, color=color, alpha=alpha, s=4, linewidths=0, zorder=3)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)

# ── Fig 1: Mean residual PSI ──────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(5, 4))
tick_labels = ["Rest", f"Bottom 2%\n(≤{thr:.1f} kcal/mol)"]
jitter_boxplot(
    ax1,
    groups=[rest["signed_residual"].values, bot["signed_residual"].values],
    tick_labels=tick_labels,
    ylabel="Residual PSI",
    title=(f"Upstream exon vs variable exon — residual PSI\n"
           f"mean bot={bot['signed_residual'].mean():+.4f}, "
           f"rest={rest['signed_residual'].mean():+.4f}, "
           f"p={p_signed:.2e}"),
)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
fig1.suptitle("RNAduplex — fixed upstream exon vs variable exon\n"
              "mean residual PSI: bottom 2% MFE vs rest",
              fontsize=10, y=1.02)
fig1.tight_layout()
p1 = OUT / "rnaduplex_upstream_exon_signed_residual.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p1.name}")

# ── Fig 2: Variance of residual PSI ──────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(4, 4))
var_rest = np.var(rest["signed_residual"].values)
var_bot  = np.var(bot["signed_residual"].values)
stars    = ("***" if p_levene < 0.001 else
            "**"  if p_levene < 0.01  else
            "*"   if p_levene < 0.05  else "ns")
tick_labels = ["Rest", f"Bottom 2%\n(≤{thr:.1f})"]
ymax = max(var_rest, var_bot)
ax2.bar([0, 1], [var_rest, var_bot], color=COLORS, width=0.5)
ax2.text(0.5, ymax * 1.05, f"{stars}\np={p_levene:.1e}",
         ha="center", va="bottom", fontsize=8)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(tick_labels, fontsize=9)
ax2.set_ylabel("Variance of residual PSI", fontsize=9)
ax2.set_title("Upstream exon vs variable exon\nresidual PSI variance (Levene)", fontsize=9)
ax2.set_ylim(0, ymax * 1.25)
fig2.suptitle("RNAduplex — fixed upstream exon vs variable exon\n"
              "residual PSI variance: bottom 2% MFE vs rest",
              fontsize=10, y=1.02)
fig2.tight_layout()
p2 = OUT / "rnaduplex_upstream_exon_psi_variance.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p2.name}")
