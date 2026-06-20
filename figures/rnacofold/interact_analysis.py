"""
Exon-intron interaction analysis.

For each of the 4 chunks, split exons into top 2% (strongest interactors)
vs rest by n_pairs, then compare:
  (a) PSI distribution  — KL divergence + overlaid histograms
  (b) Model residual    — |predicted_PSI - true_PSI|
  (c) Predicted variance
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ── Load & merge ──────────────────────────────────────────────────────────────

ann   = pd.read_csv(BASE / "data/test_annotated.csv")
cofold = pd.read_csv(BASE / "data/test_rnacofold_interact.csv")
df = ann.merge(cofold, on="exon", how="inner")
df = df[
    df["PSI"].notna() &
    df["predicted_PSI_flank_150_30_unc"].notna() &
    df["var_flank_150_30_unc"].notna()
].reset_index(drop=True)

df["residual"]        = (df["predicted_PSI_flank_150_30_unc"] - df["PSI"]).abs()
df["signed_residual"] =  df["predicted_PSI_flank_150_30_unc"] - df["PSI"]
print(f"{len(df):,} exons with complete data")

CHUNKS    = ["up_far", "up_mid", "up_near", "down"]
THRESHOLD = 0.98   # top 2%

PSI_BINS  = np.linspace(0, 1, 21)   # 20 equal-width bins
PSEUDO    = 1e-3                      # avoid log(0) in KL

def kl_div(p, q):
    p = p + PSEUDO;  p /= p.sum()
    q = q + PSEUDO;  q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'chunk':<10} {'threshold':>10} {'n_top':>7}  "
      f"{'resid_top':>10} {'resid_rest':>10}  "
      f"{'var_top':>10} {'var_rest':>10}  "
      f"{'KL(top||rest)':>14}  {'MWU p(resid)':>13}  {'MWU p(var)':>11}")

rows = []
for chunk in CHUNKS:
    col   = f"n_pairs_{chunk}"
    thr   = df[col].quantile(THRESHOLD)
    top   = df[df[col] >= thr]
    rest  = df[df[col] <  thr]

    # KL on PSI distribution
    p_top,  _ = np.histogram(top["PSI"],  bins=PSI_BINS)
    p_rest, _ = np.histogram(rest["PSI"], bins=PSI_BINS)
    kl = kl_div(p_top.astype(float), p_rest.astype(float))

    # Mann-Whitney U (one-sided: top > rest)
    _, p_resid  = mannwhitneyu(top["residual"],  rest["residual"],  alternative="greater")
    _, p_var    = mannwhitneyu(top["var_flank_150_30_unc"],
                                rest["var_flank_150_30_unc"], alternative="greater")
    # Two-sided for signed residual (bias can go either way)
    from scipy.stats import ttest_ind
    _, p_signed = ttest_ind(top["signed_residual"], rest["signed_residual"])

    rows.append(dict(chunk=chunk, threshold=thr, n_top=len(top),
                     resid_top=top["residual"].mean(),
                     resid_rest=rest["residual"].mean(),
                     var_top=top["var_flank_150_30_unc"].mean(),
                     var_rest=rest["var_flank_150_30_unc"].mean(),
                     signed_top=top["signed_residual"].mean(),
                     signed_rest=rest["signed_residual"].mean(),
                     kl=kl, p_resid=p_resid, p_var=p_var,
                     p_signed=p_signed))

    print(f"{chunk:<10} {thr:>10.0f} {len(top):>7}  "
          f"{top['residual'].mean():>10.4f} {rest['residual'].mean():>10.4f}  "
          f"{top['var_flank_150_30_unc'].mean():>10.4f} "
          f"{rest['var_flank_150_30_unc'].mean():>10.4f}  "
          f"{kl:>14.4f}  {p_resid:>13.2e}  {p_var:>11.2e}")

# ── Helpers ───────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)

def jitter_boxplot(ax, groups, colors, tick_labels, ylabel, title, n_jitter=400):
    """Boxplot with jittered strip overlay on a sample of points."""
    bp = ax.boxplot(
        groups,
        tick_labels=tick_labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        widths=0.45,
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.6)

    for j, (vals, color) in enumerate(zip(groups, colors)):
        sample = rng.choice(vals, size=min(n_jitter, len(vals)), replace=False)
        x = rng.uniform(j + 0.75, j + 1.25, size=len(sample))
        ax.scatter(x, sample, color=color, alpha=0.25, s=6, linewidths=0,
                   zorder=3)

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)


# ── Fig 1: PSI distributions ──────────────────────────────────────────────────

fig1, axes1 = plt.subplots(len(CHUNKS), 1, figsize=(6, 4 * len(CHUNKS)))
for i, (chunk, row) in enumerate(zip(CHUNKS, rows)):
    col  = f"n_pairs_{chunk}"
    thr  = row["threshold"]
    top  = df[df[col] >= thr]
    rest = df[df[col] <  thr]
    ax   = axes1[i]
    ax.hist(rest["PSI"], bins=PSI_BINS, density=True,
            alpha=0.6, color="#92c5de", label=f"Rest (n={len(rest):,})")
    ax.hist(top["PSI"],  bins=PSI_BINS, density=True,
            alpha=0.7, color="#d6604d",
            label=f"Top 2% (n={len(top):,}, ≥{thr:.0f} pairs)")
    ax.set_xlabel("True PSI", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(f"{chunk} — PSI distribution  (KL={row['kl']:.3f})", fontsize=9)
    ax.legend(fontsize=7)
fig1.suptitle("PSI distributions: top 2% interactors vs rest", fontsize=10)
fig1.tight_layout()
p1 = OUT / "interact_psi.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p1}")

# ── Fig 2: Absolute residuals ─────────────────────────────────────────────────

fig2, axes2 = plt.subplots(len(CHUNKS), 1, figsize=(6, 4 * len(CHUNKS)))
for i, (chunk, row) in enumerate(zip(CHUNKS, rows)):
    col  = f"n_pairs_{chunk}"
    thr  = row["threshold"]
    top  = df[df[col] >= thr]
    rest = df[df[col] <  thr]
    jitter_boxplot(
        axes2[i],
        groups=[rest["residual"].values, top["residual"].values],
        colors=["#92c5de", "#d6604d"],
        tick_labels=["Rest", f"Top 2%\n(≥{thr:.0f})"],
        ylabel="Absolute residual",
        title=f"{chunk} — Residual  (p={row['p_resid']:.2e})",
    )
fig2.suptitle("Absolute residuals: top 2% interactors vs rest", fontsize=10)
fig2.tight_layout()
p2 = OUT / "interact_residual.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p2}")

# ── Fig 3: Predicted variance ─────────────────────────────────────────────────

fig3, axes3 = plt.subplots(len(CHUNKS), 1, figsize=(6, 4 * len(CHUNKS)))
for i, (chunk, row) in enumerate(zip(CHUNKS, rows)):
    col  = f"n_pairs_{chunk}"
    thr  = row["threshold"]
    top  = df[df[col] >= thr]
    rest = df[df[col] <  thr]
    jitter_boxplot(
        axes3[i],
        groups=[rest["var_flank_150_30_unc"].values, top["var_flank_150_30_unc"].values],
        colors=["#92c5de", "#d6604d"],
        tick_labels=["Rest", f"Top 2%\n(≥{thr:.0f})"],
        ylabel="Predicted variance",
        title=f"{chunk} — Variance  (p={row['p_var']:.2e})",
    )
fig3.suptitle("Predicted variance: top 2% interactors vs rest", fontsize=10)
fig3.tight_layout()
p3 = OUT / "interact_variance.png"
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p3}")

# ── Fig 4: True PSI distribution boxplot ─────────────────────────────────────

from scipy.stats import levene as _levene

var_top_psi  = []
var_rest_psi = []
levene_ps    = []
for chunk in CHUNKS:
    col  = f"n_pairs_{chunk}"
    thr  = df[col].quantile(THRESHOLD)
    top  = df[df[col] >= thr]["signed_residual"].values
    rest = df[df[col] <  thr]["signed_residual"].values
    var_top_psi.append(np.var(top))
    var_rest_psi.append(np.var(rest))
    _, p = _levene(top, rest)
    levene_ps.append(p)

x     = np.arange(len(CHUNKS))
width = 0.35
fig_pv, ax_pv = plt.subplots(figsize=(8, 4))
b1 = ax_pv.bar(x - width/2, var_rest_psi, width, color="#92c5de", label="Rest")
b2 = ax_pv.bar(x + width/2, var_top_psi,  width, color="#d6604d", label="Top 2%")
for xi, p in zip(x, levene_ps):
    ymax = max(var_rest_psi[xi], var_top_psi[xi])
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax_pv.text(xi, ymax + 0.0002, f"{stars}\np={p:.1e}", ha="center", va="bottom",
               fontsize=7)
ax_pv.set_xticks(x)
ax_pv.set_xticklabels(CHUNKS, fontsize=9)
ax_pv.set_ylabel("Variance of residual PSI", fontsize=9)
ax_pv.set_title("Residual PSI variance: top 2% interactors vs rest (Levene test)", fontsize=10)
ax_pv.legend(fontsize=9)
fig_pv.tight_layout()
p_psi_box = OUT / "interact_psi_variance.png"
fig_pv.savefig(p_psi_box, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p_psi_box}")

# ── Fig 4: Signed residual boxplot ────────────────────────────────────────────

fig4, axes4 = plt.subplots(len(CHUNKS), 1, figsize=(6, 4 * len(CHUNKS)))
for i, (chunk, row) in enumerate(zip(CHUNKS, rows)):
    col  = f"n_pairs_{chunk}"
    thr  = row["threshold"]
    top  = df[df[col] >= thr]
    rest = df[df[col] <  thr]
    jitter_boxplot(
        axes4[i],
        groups=[rest["signed_residual"].values, top["signed_residual"].values],
        colors=["#92c5de", "#d6604d"],
        tick_labels=["Rest", f"Top 2%\n(≥{thr:.0f})"],
        ylabel="Residual PSI",
        title=(f"{chunk} — "
               f"mean top={row['signed_top']:+.4f}, rest={row['signed_rest']:+.4f}, "
               f"p={row['p_signed']:.2e}"),
    )
    axes4[i].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
fig4.suptitle("Difference in mean residual PSI: statistical significance", fontsize=10)
fig4.tight_layout()
p4 = OUT / "interact_signed_residual.png"
fig4.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p4}")

# ── Fig 5: Signed residual histograms (shape of error distribution) ───────────

SERR_BINS = np.linspace(-0.8, 0.8, 41)

fig5, axes5 = plt.subplots(len(CHUNKS), 1, figsize=(6, 4 * len(CHUNKS)))
for i, (chunk, row) in enumerate(zip(CHUNKS, rows)):
    col  = f"n_pairs_{chunk}"
    thr  = row["threshold"]
    top  = df[df[col] >= thr]
    rest = df[df[col] <  thr]
    ax   = axes5[i]
    ax.hist(rest["signed_residual"], bins=SERR_BINS, density=True,
            alpha=0.6, color="#92c5de", label=f"Rest (n={len(rest):,})")
    ax.hist(top["signed_residual"],  bins=SERR_BINS, density=True,
            alpha=0.7, color="#d6604d",
            label=f"Top 2% (n={len(top):,}, ≥{thr:.0f} pairs)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(row["signed_top"],  color="#d6604d", linewidth=1.5,
               linestyle="-", label=f"mean top={row['signed_top']:+.4f}")
    ax.axvline(row["signed_rest"], color="#2166ac", linewidth=1.5,
               linestyle="-", label=f"mean rest={row['signed_rest']:+.4f}")
    ax.set_xlabel("Residual PSI", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(f"{chunk}  (p={row['p_signed']:.2e})", fontsize=9)
    ax.legend(fontsize=7)
fig5.suptitle("Difference in mean residual PSI: statistical significance", fontsize=10)
fig5.tight_layout()
p5 = OUT / "interact_signed_hist.png"
fig5.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p5}")
