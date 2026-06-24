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

# ── Jitter helper ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)

def jitter_boxplot(ax, groups, colors, tick_labels, ylabel, title):
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
        # scale alpha so visual density is similar regardless of group size
        alpha = min(0.4, 200 / len(vals))
        x = rng.uniform(j + 0.75, j + 1.25, size=len(vals))
        ax.scatter(x, vals, color=color, alpha=alpha, s=4, linewidths=0, zorder=3)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)


COLORS = ["#92c5de", "#d6604d"]

from scipy.stats import levene as _levene

ANALYSES = [
    ("mfe_intrinsic",  "RNAcofold MFE",  "rnacofold"),
    ("mfe_rnaduplex",  "RNAduplex MFE",  "rnaduplex"),
]

for col, label, slug in ANALYSES:
    thr  = df[col].quantile(0.02)          # bottom 2% = most negative = most stable
    bot  = df[df[col] <= thr]
    rest = df[df[col] >  thr]

    _, p_signed = ttest_ind(bot["signed_residual"].values, rest["signed_residual"].values)
    _, p_levene = _levene(bot["signed_residual"].values,   rest["signed_residual"].values)
    stars = "***" if p_levene < 0.001 else ("**" if p_levene < 0.01 else ("*" if p_levene < 0.05 else "ns"))

    print(f"\n{label}  threshold={thr:.1f}  bot n={len(bot):,}  rest n={len(rest):,}")
    print(f"  t-test (mean signed resid): p={p_signed:.2e}")
    print(f"  Levene (var  signed resid): p={p_levene:.2e}")

    TICK_LBLS = ["Rest", f"Bottom 2%\n(≤{thr:.1f})"]

    # ── Mean residual PSI ─────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    jitter_boxplot(
        ax1,
        groups=[rest["signed_residual"].values, bot["signed_residual"].values],
        colors=COLORS,
        tick_labels=TICK_LBLS,
        ylabel="Residual PSI",
        title=(f"Intra-exon {label} — residual PSI\n"
               f"mean bot={bot['signed_residual'].mean():+.4f}, "
               f"rest={rest['signed_residual'].mean():+.4f}, p={p_signed:.2e}"),
    )
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    fig1.tight_layout()
    p1 = OUT / f"intrinsic_{slug}_signed_residual.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p1.name}")

    # ── Variance of residual PSI ──────────────────────────────────────────────
    var_rest = np.var(rest["signed_residual"].values)
    var_bot  = np.var(bot["signed_residual"].values)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.bar([0, 1], [var_rest, var_bot], color=COLORS, width=0.5)
    ymax = max(var_rest, var_bot)
    ax2.text(0.5, ymax * 1.05, f"{stars}\np={p_levene:.1e}",
             ha="center", va="bottom", fontsize=8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(TICK_LBLS, fontsize=9)
    ax2.set_ylabel("Variance of residual PSI", fontsize=9)
    ax2.set_title(f"Intra-exon {label}\nresidual PSI variance (Levene test)", fontsize=9)
    ax2.set_ylim(0, ymax * 1.25)
    fig2.tight_layout()
    p2 = OUT / f"intrinsic_{slug}_psi_variance.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p2.name}")
