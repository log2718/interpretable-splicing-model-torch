"""
Residual analysis: positional weighted structure score vs prediction error.

Data: test_flank_150_30.csv merged with kl from test_annotated.csv
Sequence layout: 250 nt total
  - upstream flank: 0–149
  - exon:           150–219 (70 nt)
  - downstream flank: 220–249
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from pathlib import Path

OUT = Path("figures/residual_analysis")
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 10
SEQ_LEN = 250
EXON_START = 150
EXON_END = 220   # exclusive

REGIONS = {
    "upstream_flank":   (0,   150),
    "exon":             (150, 220),
    "downstream_flank": (220, 250),
    "near_3ss":         (127, 150),   # 23 bp before exon start
    "near_5ss":         (220, 229),   # 9 bp after exon end
}

# ── helpers ──────────────────────────────────────────────────────────────────

BP_WEIGHTS = {
    frozenset("GC"): 3,
    frozenset("AU"): 2,
    frozenset("GU"): 1,
}

def get_pair_map(dot_bracket: str) -> dict[int, int]:
    """Return {i: j} for each paired position."""
    stack = []
    pairs = {}
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j
    return pairs


def bp_weight(i: int, j: int, seq: str) -> float:
    b1, b2 = seq[i].upper(), seq[j].upper()
    return BP_WEIGHTS.get(frozenset((b1, b2)), 0)


def weighted_structure_scores(dot_bracket: str, seq: str, window: int = WINDOW) -> np.ndarray:
    """
    For each position p, sum bp weights for paired bases in the window
    centred on p: [p - half, p + half), clamped to sequence boundaries.
    Returns array of length len(dot_bracket).
    """
    pairs = get_pair_map(dot_bracket)
    n = len(dot_bracket)
    half = window // 2
    scores = np.zeros(n)
    for p in range(n):
        start = max(0, p - half)
        end = min(n, p - half + window)
        scores[p] = sum(
            bp_weight(i, pairs[i], seq)
            for i in range(start, end)
            if dot_bracket[i] in "()" and i in pairs
        )
    return scores


# ── load and merge data ───────────────────────────────────────────────────────

print("Loading data...")
df_main = pd.read_csv("data/test_flank_150_30.csv")
df_ann  = pd.read_csv("data/test_annotated.csv")[["exon", "kl_flank_150_30"]]

df = df_main.merge(df_ann, on="exon", how="inner")
df = df.drop(columns=["predicted_mfe"])   # shorter-seq carry-over; use predicted_MFE instead
df = df.rename(columns={
    "PSI": "true_PSI",
    "predicted_secondary_struct": "structure",
    "kl_flank_150_30": "kl",
    "predicted_MFE": "predicted_mfe",
})
df["residual"] = df["predicted_PSI"] - df["true_PSI"]

# Filter rows with valid structure and sequence of expected length
mask = (
    df["structure"].notna() &
    df["model_sequence"].notna() &
    (df["structure"].str.len() == SEQ_LEN) &
    (df["model_sequence"].str.len() == SEQ_LEN)
)
df = df[mask].reset_index(drop=True)
print(f"  {len(df)} examples after filtering")

# ── compute positional scores ─────────────────────────────────────────────────

print("Computing positional structure scores (this takes a while)...")
score_matrix = np.zeros((len(df), SEQ_LEN))
for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"  {idx}/{len(df)}")
    score_matrix[idx] = weighted_structure_scores(row["structure"], row["model_sequence"])

print("  done.")

# Save per-example positional scores (rows=examples, cols=positions 0..249)
score_df = pd.DataFrame(score_matrix, columns=[f"pos_{i}" for i in range(SEQ_LEN)])
score_df.insert(0, "exon", df["exon"].values)
score_df.to_csv(OUT / "positional_scores.csv", index=False)
print(f"  positional scores saved to {OUT}/positional_scores.csv")

# Regional mean scores per example
for name, (lo, hi) in REGIONS.items():
    df[f"score_{name}"] = score_matrix[:, lo:hi].mean(axis=1)

residual = df["residual"].values
abs_residual = np.abs(residual)
mfe = df["predicted_mfe"].values

# ── plot helpers ──────────────────────────────────────────────────────────────

BLUES = plt.cm.Blues
REGION_LABELS = {
    "upstream_flank":   "Upstream flank (0–149)",
    "exon":             "Exon (150–219)",
    "downstream_flank": "Downstream flank (220–249)",
    "near_3ss":         "Near 3′ SS (127–149)",
    "near_5ss":         "Near 5′ SS (220–228)",
}

N_BINS = 20


def equal_freq_bins(values, n_bins):
    """Return bin indices using equal-frequency (quantile) binning."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, quantiles)
    edges = np.unique(edges)
    labels = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return labels, edges


# ── Figure 1: mean residual vs binned score, per region ──────────────────────

print("Plotting figure 1: mean residual vs binned structure score...")
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, (region, label) in zip(axes, REGION_LABELS.items()):
    scores = df[f"score_{region}"].values
    bin_ids, edges = equal_freq_bins(scores, N_BINS)
    bin_mids, means, = [], []
    for b in range(N_BINS):
        sel = bin_ids == b
        if sel.sum() == 0:
            continue
        bin_mids.append(scores[sel].mean())
        means.append(residual[sel].mean())
    bin_mids = np.array(bin_mids)
    means = np.array(means)
    ax.bar(range(len(bin_mids)), means, color=BLUES(0.6), edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(label, fontsize=8)
    ax.set_xlabel("Score bin", fontsize=7)
    ax.tick_params(axis="x", labelbottom=False)

axes[0].set_ylabel("Mean residual (pred − true)", fontsize=8)
fig.suptitle("Mean Residual vs Binned Structure Score by Region", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig1_mean_residual_vs_score.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 2: variance of residual vs binned score, per region ───────────────

print("Plotting figure 2: variance of residual vs binned score...")
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, (region, label) in zip(axes, REGION_LABELS.items()):
    scores = df[f"score_{region}"].values
    bin_ids, _ = equal_freq_bins(scores, N_BINS)
    bin_mids, variances = [], []
    for b in range(N_BINS):
        sel = bin_ids == b
        if sel.sum() < 2:
            continue
        bin_mids.append(scores[sel].mean())
        variances.append(residual[sel].var())
    bin_mids = np.array(bin_mids)
    variances = np.array(variances)
    ax.bar(range(len(bin_mids)), variances, color=BLUES(0.6), edgecolor="white", linewidth=0.3)
    ax.set_title(label, fontsize=8)
    ax.set_xlabel("Score bin", fontsize=7)
    ax.tick_params(axis="x", labelbottom=False)

axes[0].set_ylabel("Variance of residual", fontsize=8)
fig.suptitle("Residual Variance vs Binned Structure Score by Region", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig2_residual_variance_vs_score.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 3: scatter score vs residual, colored by MFE ──────────────────────

print("Plotting figure 3: scatter score vs residual colored by MFE...")
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
mfe_norm = mcolors.Normalize(vmin=np.percentile(mfe, 2), vmax=np.percentile(mfe, 98))
for ax, (region, label) in zip(axes, REGION_LABELS.items()):
    scores = df[f"score_{region}"].values
    sc = ax.scatter(scores, residual, c=mfe, cmap="Blues_r",
                    norm=mfe_norm, alpha=0.3, s=2, rasterized=True)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(label, fontsize=8)
    ax.set_xlabel("Mean structure score", fontsize=7)

axes[0].set_ylabel("Residual (pred − true)", fontsize=8)
cbar = fig.colorbar(sc, ax=axes[-1], fraction=0.05)
cbar.set_label("Predicted MFE (kcal/mol)", fontsize=7)
fig.suptitle("Structure Score vs Residual (colored by MFE)", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig3_scatter_score_vs_residual_mfe.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 3b: 2D histogram of structure score vs residual, per region ───────

print("Plotting figure 3b: 2D histogram structure score vs residual...")
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (region, label) in zip(axes, REGION_LABELS.items()):
    scores = df[f"score_{region}"].values
    h, xedges, yedges = np.histogram2d(
        scores, residual,
        bins=50,
        range=[[scores.min(), scores.max()],
               [np.percentile(residual, 1), np.percentile(residual, 99)]],
    )
    h = np.ma.masked_where(h == 0, h)
    im = ax.pcolormesh(xedges, yedges, h.T,
                       cmap="Blues",
                       norm=mcolors.LogNorm(vmin=0.6, vmax=h.max()))
    ax.axhline(0, color="tomato", linewidth=0.8, linestyle="--")
    ax.set_title(label, fontsize=8)
    ax.set_xlabel("Mean structure score", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Count", fontsize=6)

axes[0].set_ylabel("Residual (pred − true)", fontsize=8)
fig.suptitle("2D Histogram: Structure Score vs Residual by Region", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig3b_2d_hist_score_vs_residual.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 4: positional Pearson correlation heatmap ─────────────────────────

print("Computing positional Pearson correlations...")
pos_corr = np.zeros(SEQ_LEN)
for pos in range(SEQ_LEN):
    col = score_matrix[:, pos]
    if col.std() < 1e-10:
        pos_corr[pos] = 0.0
    else:
        r, _ = pearsonr(col, abs_residual)
        pos_corr[pos] = r

print("Plotting figure 4: positional correlation heatmap...")

# ── Shared helper for positional line plots (also used by figs 5–7) ──────────

def plot_positional_line(corr_array, ylabel, title, out_path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    positions = np.arange(SEQ_LEN)
    ax.plot(positions, corr_array, color=BLUES(0.7), linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvspan(0,   150, alpha=0.04, color="steelblue", label="Upstream flank")
    ax.axvspan(150, 220, alpha=0.10, color="navy",      label="Exon")
    ax.axvspan(220, 250, alpha=0.04, color="steelblue", label="Downstream flank")
    for x, lbl in [(127, "3′ SS start"), (229, "5′ SS end")]:
        ax.axvline(x, color="tomato", linewidth=0.9, linestyle=":", alpha=0.8)
        ax.text(x + 0.5, 0.01, lbl, rotation=90, fontsize=6,
                va="bottom", ha="left", color="tomato",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("Position along sequence (nt)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, SEQ_LEN - 1)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


plot_positional_line(
    pos_corr,
    ylabel="Pearson r (structure score, |residual|)",
    title="Positional Correlation of Structure Score with Absolute Residual",
    out_path=OUT / "fig4_positional_correlation.png",
)

# ── Figure 5: partial correlation (structure score vs |residual|, MFE partialled out) ──

print("Computing partial correlations (controlling for MFE)...")

def partial_corr(x, y, z):
    """Pearson partial correlation of x and y controlling for z."""
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return 0.0
    return (r_xy - r_xz * r_yz) / denom

pos_partial = np.zeros(SEQ_LEN)
for pos in range(SEQ_LEN):
    col = score_matrix[:, pos]
    if col.std() < 1e-10:
        pos_partial[pos] = 0.0
    else:
        pos_partial[pos] = partial_corr(col, abs_residual, mfe)

plot_positional_line(
    pos_partial,
    ylabel="Partial r (structure score, |residual| | MFE)",
    title="Positional Partial Correlation with |Residual|, Controlling for Global MFE",
    out_path=OUT / "fig5_partial_correlation.png",
)
print("  fig5 saved.")

# ── Figures 6 & 7: fig4 split by over- vs under-prediction ───────────────────

over_mask  = residual > 0
under_mask = residual < 0

print(f"Over-prediction:  {over_mask.sum()} examples")
print(f"Under-prediction: {under_mask.sum()} examples")

def pos_pearsonr_subset(mask):
    arr = np.zeros(SEQ_LEN)
    sub_abs = np.abs(residual[mask])
    for pos in range(SEQ_LEN):
        col = score_matrix[mask, pos]
        if col.std() < 1e-10:
            arr[pos] = 0.0
        else:
            r, _ = pearsonr(col, sub_abs)
            arr[pos] = r
    return arr

print("Computing correlations for over-predictions...")
pos_over  = pos_pearsonr_subset(over_mask)
print("Computing correlations for under-predictions...")
pos_under = pos_pearsonr_subset(under_mask)

# Use a shared symmetric colour range so the two plots are comparable
shared_vmin = min(pos_over.min(), pos_under.min())
shared_vmax = max(pos_over.max(), pos_under.max())

plot_positional_line(
    pos_over,
    ylabel="Pearson r (structure score, residual magnitude)",
    title=f"Positional Correlation with Residual Magnitude — Over-predictions only (n={over_mask.sum()})",
    out_path=OUT / "fig6_positional_correlation_over.png",
    vmin=shared_vmin, vmax=shared_vmax,
)
print("  fig6 saved.")

plot_positional_line(
    pos_under,
    ylabel="Pearson r (structure score, residual magnitude)",
    title=f"Positional Correlation with Residual Magnitude — Under-predictions only (n={under_mask.sum()})",
    out_path=OUT / "fig7_positional_correlation_under.png",
    vmin=shared_vmin, vmax=shared_vmax,
)
print("  fig7 saved.")

# ── Figure 8: combined over + sign-flipped under ─────────────────────────────
# pos_over:  r(structure, |residual|) for residual>0  →  positive r = more structure → larger over-prediction
# -pos_under: negate r(structure, |residual|) for residual<0  →  positive r = more structure → less negative (higher value)
# Both now share the same directional meaning: positive r = more structure → higher residual value

print("Plotting figure 8: combined directional correlation...")
# Average over-prediction r with sign-flipped under-prediction r.
# Both carry the same directional meaning: positive = more structure → higher residual value.
pos_combined = (pos_over + (-pos_under)) / 2

fig, ax = plt.subplots(figsize=(14, 3.5))
positions = np.arange(SEQ_LEN)

ax.plot(positions, pos_combined, color=BLUES(0.7), linewidth=1.2)
ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
ax.axvspan(0,   150, alpha=0.04, color="steelblue", label="Upstream flank")
ax.axvspan(150, 220, alpha=0.10, color="navy",      label="Exon")
ax.axvspan(220, 250, alpha=0.04, color="steelblue", label="Downstream flank")
for x, lbl in [(127, "3′ SS start"), (229, "5′ SS end")]:
    ax.axvline(x, color="tomato", linewidth=0.9, linestyle=":", alpha=0.8)
    ax.text(x + 0.5, 0.01, lbl, rotation=90, fontsize=6,
            va="bottom", ha="left", color="tomato",
            transform=ax.get_xaxis_transform())

ax.set_xlabel("Position along sequence (nt)", fontsize=9)
ax.set_ylabel("Correlation between structure and residual value", fontsize=9)
ax.set_title("Correlation between Structure and Residual Value by Position", fontsize=10)
ax.set_xlim(0, SEQ_LEN - 1)
ax.legend(loc="upper left", fontsize=6, framealpha=0.5)
plt.tight_layout()
plt.savefig(OUT / "fig8_combined_directional_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig8 saved.")

print(f"\nAll figures saved to {OUT}/")
