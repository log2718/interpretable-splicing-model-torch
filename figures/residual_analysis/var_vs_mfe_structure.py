"""
Plot predicted variance (var_vienna60_unc) against:
  1. Binned MFE (MFE_vienna60)
  2. Binned mean weighted structure score (computed from struct_vienna60)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

OUT = Path("figures/residual_analysis")
BLUES = plt.cm.Blues
N_BINS = 20

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv("data/test_annotated.csv")
df = df[df["var_vienna60_unc"].notna() &
        df["MFE_vienna60"].notna() &
        df["struct_vienna60"].notna() &
        df["model_seq_vienna60"].notna()].reset_index(drop=True)
print(f"{len(df):,} examples")

var = df["var_vienna60_unc"].values
mfe = df["MFE_vienna60"].values

# ── Weighted structure score ───────────────────────────────────────────────────

BP_WEIGHTS = {frozenset("GC"): 3, frozenset("AU"): 2, frozenset("GU"): 1}

def get_pair_map(dot_bracket):
    stack, pairs = [], {}
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    return pairs

def bp_weight(i, j, seq):
    return BP_WEIGHTS.get(frozenset((seq[i].upper(), seq[j].upper())), 0)

def mean_struct_score(dot_bracket, seq):
    """Sum of bp weights across all pairs."""
    pairs = get_pair_map(dot_bracket)
    return sum(bp_weight(i, pairs[i], seq)
               for i in pairs if dot_bracket[i] == "(")

print("Computing structure scores...")
struct_scores = np.array([
    mean_struct_score(row["struct_vienna60"], row["model_seq_vienna60"])
    for _, row in df.iterrows()
])
print("  done.")

# ── Binning helper ────────────────────────────────────────────────────────────

def equal_freq_bins(values, n_bins):
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(values, quantiles))
    labels = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return labels

def bin_stats(x_vals, y_vals, n_bins=N_BINS):
    """Returns (bin_mids, means, stds) using equal-frequency bins on x."""
    bin_ids = equal_freq_bins(x_vals, n_bins)
    mids, means, stds = [], [], []
    for b in range(n_bins):
        sel = bin_ids == b
        if sel.sum() < 2:
            continue
        mids.append(x_vals[sel].mean())
        means.append(y_vals[sel].mean())
        stds.append(y_vals[sel].std())
    return np.array(mids), np.array(means), np.array(stds)

# ── Figure 1: var vs binned MFE ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
mids, means, _ = bin_stats(mfe, var)
ax.plot(mids, means, color=BLUES(0.7), linewidth=1.5, marker="o", markersize=4)
ax.set_xlabel("MFE (kcal/mol)", fontsize=9)
ax.set_ylabel("Mean predicted variance", fontsize=9)
ax.set_title("Predicted Variance vs MFE (ViennaRNA 60°C)", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "var_vs_mfe.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved var_vs_mfe.png")

# ── Figure 2: var vs binned structure score ───────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
mids2, means2, _ = bin_stats(struct_scores, var)
ax.plot(mids2, means2, color=BLUES(0.7), linewidth=1.5, marker="o", markersize=4)
ax.set_xlabel("Mean weighted structure score", fontsize=9)
ax.set_ylabel("Mean predicted variance", fontsize=9)
ax.set_title("Predicted Variance vs Structure Score (ViennaRNA 60°C)", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "var_vs_structure_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved var_vs_structure_score.png")
