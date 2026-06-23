"""
Positive-control: intra-exon interaction analysis.

RNAcofold(first 35 nt & last 35 nt of exon) → n_pairs_intrinsic.
Split exons into top 2% (strongest intra-exon interactors) vs rest.
Compare:
  (a) Absolute residual  |predicted_PSI - true_PSI|
  (b) Predicted variance
  (c) Signed residual (residual PSI)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ── Load & merge ──────────────────────────────────────────────────────────────

ann      = pd.read_csv(BASE / "data/test_annotated.csv")
intrinsic = pd.read_csv(BASE / "data/test_rnacofold_intrinsic.csv")
df = ann.merge(intrinsic, on="exon", how="inner")
df = df[
    df["PSI"].notna() &
    df["predicted_PSI_flank_150_30_unc"].notna() &
    df["var_flank_150_30_unc"].notna()
].reset_index(drop=True)

df["residual"]        = (df["predicted_PSI_flank_150_30_unc"] - df["PSI"]).abs()
df["signed_residual"] =  df["predicted_PSI_flank_150_30_unc"] - df["PSI"]
print(f"{len(df):,} exons with complete data")

COL       = "n_pairs_intrinsic"
THRESHOLD = 0.98

thr  = df[COL].quantile(THRESHOLD)
top  = df[df[COL] >= thr]
rest = df[df[COL] <  thr]
print(f"Threshold (top 2%): {thr:.0f} pairs")
print(f"  top n={len(top):,}  rest n={len(rest):,}")

_, p_resid  = mannwhitneyu(top["residual"],  rest["residual"],  alternative="greater")
_, p_var    = mannwhitneyu(top["var_flank_150_30_unc"],
                            rest["var_flank_150_30_unc"], alternative="greater")
_, p_signed = ttest_ind(top["signed_residual"], rest["signed_residual"])

print(f"MWU p (|residual| top>rest): {p_resid:.2e}")
print(f"MWU p (variance   top>rest): {p_var:.2e}")
print(f"t-test p (signed residual):  {p_signed:.2e}")
print(f"  mean residual  top={top['signed_residual'].mean():+.4f}  "
      f"rest={rest['signed_residual'].mean():+.4f}")
print(f"  mean variance  top={top['var_flank_150_30_unc'].mean():.4f}  "
      f"rest={rest['var_flank_150_30_unc'].mean():.4f}")

# ── Jitter helper ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)

def jitter_boxplot(ax, groups, colors, tick_labels, ylabel, title, n_jitter=400):
    bp = ax.boxplot(
        groups,
        tick_labels=tick_labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        widths=0.45,
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color); box.set_alpha(0.6)
    for j, (vals, color) in enumerate(zip(groups, colors)):
        sample = rng.choice(vals, size=min(n_jitter, len(vals)), replace=False)
        x = rng.uniform(j + 0.75, j + 1.25, size=len(sample))
        ax.scatter(x, sample, color=color, alpha=0.25, s=6, linewidths=0, zorder=3)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)


COLORS     = ["#92c5de", "#d6604d"]
TICK_LBLS  = ["Rest", f"Top 2%\n(≥{thr:.0f} pairs)"]

# ── Fig 1: Absolute residual ──────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(5, 4))
jitter_boxplot(
    ax1,
    groups=[rest["residual"].values, top["residual"].values],
    colors=COLORS,
    tick_labels=TICK_LBLS,
    ylabel="Absolute residual",
    title=f"Intra-exon interaction — absolute residual\n(MWU p={p_resid:.2e})",
)
fig1.tight_layout()
p1 = OUT / "intrinsic_residual.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p1}")

# ── Fig 2: Predicted variance ─────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(5, 4))
jitter_boxplot(
    ax2,
    groups=[rest["var_flank_150_30_unc"].values, top["var_flank_150_30_unc"].values],
    colors=COLORS,
    tick_labels=TICK_LBLS,
    ylabel="Predicted variance",
    title=f"Intra-exon interaction — predicted variance\n(MWU p={p_var:.2e})",
)
fig2.tight_layout()
p2 = OUT / "intrinsic_variance.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p2}")

# ── Fig 3: Signed residual ────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(5, 4))
jitter_boxplot(
    ax3,
    groups=[rest["signed_residual"].values, top["signed_residual"].values],
    colors=COLORS,
    tick_labels=TICK_LBLS,
    ylabel="Residual PSI",
    title=(f"Intra-exon interaction — residual PSI\n"
           f"mean top={top['signed_residual'].mean():+.4f}, "
           f"rest={rest['signed_residual'].mean():+.4f}, "
           f"p={p_signed:.2e}"),
)
ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
fig3.tight_layout()
p3 = OUT / "intrinsic_signed_residual.png"
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p3}")
