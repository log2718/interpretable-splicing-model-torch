"""
Sanity check: do exons with the most stable intra-exon structure (bottom 2% RNAfold MFE)
show higher residuals or variance?

Uses MFE_flank_150_30 from test_annotated.csv — RNAfold MFE of the full 250-nt window
(fixed flanks + exon). Since flanks are identical for all exons, all variance in MFE
comes from the exon sequence itself.

Bottom 2% = most negative MFE = most stably folded exons.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, levene as levene_test

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(BASE / "data/test_annotated.csv")
df = df[
    df["PSI"].notna() &
    df["predicted_PSI_flank_150_30_unc"].notna() &
    df["var_flank_150_30_unc"].notna() &
    df["MFE_flank_150_30"].notna()
].reset_index(drop=True)

df["residual"]        = (df["predicted_PSI_flank_150_30_unc"] - df["PSI"]).abs()
df["signed_residual"] =  df["predicted_PSI_flank_150_30_unc"] - df["PSI"]
print(f"{len(df):,} exons with complete data")

COL       = "MFE_flank_150_30"
THRESHOLD = 0.02   # bottom 2% = most negative MFE

thr   = df[COL].quantile(THRESHOLD)
bot   = df[df[COL] <= thr]   # most stable structures
rest  = df[df[COL] >  thr]
print(f"Bottom 2% MFE threshold: {thr:.1f} kcal/mol")
print(f"  bottom n={len(bot):,}   rest n={len(rest):,}")

_, p_resid  = mannwhitneyu(bot["residual"],  rest["residual"],  alternative="greater")
_, p_var    = mannwhitneyu(bot["var_flank_150_30_unc"],
                            rest["var_flank_150_30_unc"], alternative="greater")
_, p_signed = ttest_ind(bot["signed_residual"], rest["signed_residual"])
_, p_levene = levene_test(bot["signed_residual"].values, rest["signed_residual"].values)

print(f"MWU p (|residual|  bot>rest): {p_resid:.2e}")
print(f"MWU p (pred var    bot>rest): {p_var:.2e}")
print(f"t-test p (mean signed resid): {p_signed:.2e}")
print(f"Levene p (var of resid):      {p_levene:.2e}")

# ── Jitter helper ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
COLORS    = ["#92c5de", "#d6604d"]
TICK_LBLS = ["Rest", f"Bottom 2%\n(≤{thr:.0f} kcal/mol)"]

def jitter_boxplot(ax, groups, ylabel, title):
    bp = ax.boxplot(groups, tick_labels=TICK_LBLS, patch_artist=True,
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

# ── Fig 1: Absolute residual ──────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(5, 4))
jitter_boxplot(ax1,
    [rest["residual"].values, bot["residual"].values],
    "Absolute residual",
    f"RNAfold MFE — absolute residual\n(MWU p={p_resid:.2e})")
fig1.tight_layout()
fig1.savefig(OUT / "mfe_residual.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved mfe_residual.png")

# ── Fig 2: Signed residual ────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(5, 4))
jitter_boxplot(ax2,
    [rest["signed_residual"].values, bot["signed_residual"].values],
    "Residual PSI",
    (f"RNAfold MFE — residual PSI\n"
     f"mean bot={bot['signed_residual'].mean():+.4f}, "
     f"rest={rest['signed_residual'].mean():+.4f}, p={p_signed:.2e}"))
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
fig2.tight_layout()
fig2.savefig(OUT / "mfe_signed_residual.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved mfe_signed_residual.png")

# ── Fig 3: Predicted variance ─────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(5, 4))
jitter_boxplot(ax3,
    [rest["var_flank_150_30_unc"].values, bot["var_flank_150_30_unc"].values],
    "Predicted variance",
    f"RNAfold MFE — predicted variance\n(MWU p={p_var:.2e})")
fig3.tight_layout()
fig3.savefig(OUT / "mfe_variance.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved mfe_variance.png")

# ── Fig 4: Variance of residual PSI ──────────────────────────────────────────

var_rest = np.var(rest["signed_residual"].values)
var_bot  = np.var(bot["signed_residual"].values)
stars = "***" if p_levene < 0.001 else ("**" if p_levene < 0.01 else ("*" if p_levene < 0.05 else "ns"))

fig4, ax4 = plt.subplots(figsize=(4, 4))
ax4.bar([0, 1], [var_rest, var_bot], color=COLORS, width=0.5)
ymax = max(var_rest, var_bot)
ax4.text(0.5, ymax * 1.05, f"{stars}\np={p_levene:.1e}",
         ha="center", va="bottom", fontsize=8)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(TICK_LBLS, fontsize=9)
ax4.set_ylabel("Variance of residual PSI", fontsize=9)
ax4.set_title("RNAfold MFE\nresidual PSI variance (Levene test)", fontsize=9)
ax4.set_ylim(0, ymax * 1.25)
fig4.tight_layout()
fig4.savefig(OUT / "mfe_psi_variance.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved mfe_psi_variance.png")
