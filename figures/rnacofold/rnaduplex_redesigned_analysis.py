"""
Redesigned exon-intron interaction analysis.

Chunks (near to far upstream, then downstream):
  ss3             LEFT_FLANK_150[127:150]   23nt  3' splice site
  up_near         LEFT_FLANK_150[100:150]   50nt  (includes 3'ss, 10nt overlap with up_mid)
  up_mid          LEFT_FLANK_150[ 60:110]   50nt  (10nt overlap each side)
  up_far          LEFT_FLANK_150[ 20: 70]   50nt  (10nt overlap with up_mid)
  downstream_short RIGHT_FLANK_100[0:100]   100nt
  downstream_long  DOWNSTREAM_LONG[0:250]   250nt
  downstream_exon  DOWNSTREAM_EXON[0:141]   141nt (~853nt downstream of variable exon)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene as levene_test

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent / "redesigned"
OUT.mkdir(parents=True, exist_ok=True)

ann    = pd.read_csv(BASE / "data/test_annotated.csv")
duplex = pd.read_csv(BASE / "data/test_rnaduplex_redesigned.csv")
df = ann.merge(duplex, on="exon", how="inner")
df = df[
    df["PSI"].notna() &
    df["predicted_PSI_flank_150_30_unc"].notna() &
    df["var_flank_150_30_unc"].notna()
].reset_index(drop=True)

df["signed_residual"] = df["predicted_PSI_flank_150_30_unc"] - df["PSI"]
print(f"{len(df):,} exons with complete data")

CHUNKS = ["ss3", "up_near", "up_mid", "up_far", "up_total",
          "upstream_exon",
          "downstream_short", "downstream_long", "downstream_exon"]
COLORS = ["#92c5de", "#d6604d"]
rng    = np.random.default_rng(42)


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


rows = []
print(f"\n{'chunk':<18} {'threshold':>10} {'n_bot':>7}  "
      f"{'mean_bot':>9} {'mean_rest':>10}  "
      f"{'p_mean':>10}  {'p_levene':>10}")

for chunk in CHUNKS:
    col  = f"mfe_{chunk}"
    thr  = df[col].quantile(0.02)
    bot  = df[df[col] <= thr]
    rest = df[df[col] >  thr]

    _, p_signed = ttest_ind(bot["signed_residual"].values, rest["signed_residual"].values)
    _, p_levene = levene_test(bot["signed_residual"].values, rest["signed_residual"].values)

    rows.append(dict(chunk=chunk, thr=thr, bot=bot, rest=rest,
                     p_signed=p_signed, p_levene=p_levene))

    print(f"{chunk:<18} {thr:>10.1f} {len(bot):>7}  "
          f"{bot['signed_residual'].mean():>+9.4f} "
          f"{rest['signed_residual'].mean():>+10.4f}  "
          f"{p_signed:>10.2e}  {p_levene:>10.2e}")

# ── Fig 1: Mean residual PSI ──────────────────────────────────────────────────

fig1, axes1 = plt.subplots(len(CHUNKS), 1, figsize=(5, 4 * len(CHUNKS)))
for ax, r in zip(axes1, rows):
    thr = r["thr"]
    tick_labels = ["Rest", f"Bottom 2%\n(≤{thr:.1f} kcal/mol)"]
    jitter_boxplot(
        ax,
        groups=[r["rest"]["signed_residual"].values, r["bot"]["signed_residual"].values],
        tick_labels=tick_labels,
        ylabel="Residual PSI",
        title=(f"{r['chunk']} — residual PSI\n"
               f"mean bot={r['bot']['signed_residual'].mean():+.4f}, "
               f"rest={r['rest']['signed_residual'].mean():+.4f}, "
               f"p={r['p_signed']:.2e}"),
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
fig1.suptitle("RNAduplex redesigned chunks — mean residual PSI: bottom 2% vs rest",
              fontsize=10, y=1.02)
fig1.tight_layout()
p1 = OUT / "rnaduplex_redesigned_signed_residual.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {p1.name}")

# ── Fig 2: Variance of residual PSI ──────────────────────────────────────────

fig2, axes2 = plt.subplots(len(CHUNKS), 1, figsize=(4, 4 * len(CHUNKS)))
for ax, r in zip(axes2, rows):
    thr      = r["thr"]
    var_rest = np.var(r["rest"]["signed_residual"].values)
    var_bot  = np.var(r["bot"]["signed_residual"].values)
    stars    = ("***" if r["p_levene"] < 0.001 else
                "**"  if r["p_levene"] < 0.01  else
                "*"   if r["p_levene"] < 0.05  else "ns")
    tick_labels = ["Rest", f"Bottom 2%\n(≤{thr:.1f})"]
    ymax = max(var_rest, var_bot)
    ax.bar([0, 1], [var_rest, var_bot], color=COLORS, width=0.5)
    ax.text(0.5, ymax * 1.05, f"{stars}\np={r['p_levene']:.1e}",
            ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("Variance of residual PSI", fontsize=9)
    ax.set_title(f"{r['chunk']} — residual PSI variance (Levene)", fontsize=9)
    ax.set_ylim(0, ymax * 1.25)
fig2.suptitle("RNAduplex redesigned chunks — residual PSI variance: bottom 2% vs rest",
              fontsize=10, y=1.02)
fig2.tight_layout()
p2 = OUT / "rnaduplex_redesigned_psi_variance.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p2.name}")
