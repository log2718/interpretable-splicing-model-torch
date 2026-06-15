"""
Uncertainty model diagnostic figures — flank_150_30_unc model.

Fig 1: Bottleneck weight bar chart (56 filter weights)
Fig 2: 2D heatmap — mean variance by MFE × predicted PSI
Fig 3: 1D line plots — mean variance vs MFE and vs structure score
Fig 4: Calibration — mean |residual| vs binned predicted variance
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sys

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
from model import PNASModel

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

BLUES   = plt.cm.Blues
N_BINS  = 20
CKPT    = BASE / "checkpoints/flank_150_30_uncertainty/best_model_20260613_171626.pt"
CSV     = BASE / "data/test_annotated.csv"

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV)
df = df[
    df["var_flank_150_30_unc"].notna() &
    df["MFE_flank_150_30"].notna() &
    df["predicted_PSI_flank_150_30_unc"].notna() &
    df["PSI"].notna()
].reset_index(drop=True)
print(f"{len(df):,} examples with complete flank_150_30_unc data")

var     = df["var_flank_150_30_unc"].values
mfe     = df["MFE_flank_150_30"].values
pred    = df["predicted_PSI_flank_150_30_unc"].values
psi     = df["PSI"].values
resid   = np.abs(pred - psi)

# ── Structure score (from vienna60 columns — same exon, different flanks) ──────

BP_WEIGHTS = {frozenset("GC"): 3, frozenset("AU"): 2, frozenset("GU"): 1}

def get_pair_map(db):
    stack, pairs = [], {}
    for i, c in enumerate(db):
        if c == "(": stack.append(i)
        elif c == ")" and stack:
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs

def bp_weight(i, j, seq):
    return BP_WEIGHTS.get(frozenset((seq[i].upper(), seq[j].upper())), 0)

def struct_score(db, seq):
    pairs = get_pair_map(db)
    return sum(bp_weight(i, pairs[i], seq) for i in pairs if db[i] == "(")

struct_df = df[df["struct_vienna60"].notna() & df["model_seq_vienna60"].notna()]
print("Computing structure scores...")
struct_scores = np.array([
    struct_score(r["struct_vienna60"], r["model_seq_vienna60"])
    for _, r in struct_df.iterrows()
])
print("  done.")

# ── Binning helpers ────────────────────────────────────────────────────────────

def bin_stats(x_vals, y_vals, n=N_BINS):
    """Equal-frequency bins on x; return (midpoints, means, stds)."""
    edges = np.unique(np.percentile(x_vals, np.linspace(0, 100, n + 1)))
    edges[0] -= 1e-10; edges[-1] += 1e-10
    labels = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
    mids, means, stds = [], [], []
    for b in range(len(edges) - 1):
        sel = labels == b
        if sel.sum() < 2: continue
        mids.append(x_vals[sel].mean())
        means.append(y_vals[sel].mean())
        stds.append(y_vals[sel].std())
    return np.array(mids), np.array(means), np.array(stds)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Bottleneck weight bar chart
# ══════════════════════════════════════════════════════════════════════════════

print("Loading checkpoint for bottleneck weights...")
raw   = torch.load(CKPT, map_location="cpu", weights_only=False)
sd    = raw.get("model_state_dict", raw)
# variance_bottleneck.weight shape: (1, 56)
w = sd["variance_bottleneck.weight"].squeeze().numpy()   # (56,)

# Layout: [incl_seq(0-19), incl_struct(20-27), skip_seq(28-47), skip_struct(48-55)]
labels = (
    [f"I-seq {i}"    for i in range(20)] +
    [f"I-str {i}"    for i in range(8)]  +
    [f"S-seq {i}"    for i in range(20)] +
    [f"S-str {i}"    for i in range(8)]
)
colors = ["#2166ac" if v >= 0 else "#d6604d" for v in w]
x      = np.arange(56)

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(x, w, color=colors, width=0.8)
ax.axhline(0, color="black", linewidth=0.6)
# dashed line separates the 28 inclusion filters (left) from 28 skipping filters (right)
ax.axvline(27.5, color="gray", linewidth=1, linestyle="--")

# Region labels above the plot
ymax = ax.get_ylim()[1]
ax.text(13.5,  ax.get_ylim()[1] * 0.97, "← Inclusion filters →",
        ha="center", va="top", fontsize=8, color="gray")
ax.text(41.5,  ax.get_ylim()[1] * 0.97, "← Skipping filters →",
        ha="center", va="top", fontsize=8, color="gray")

# Group labels on x-axis: seq 0-19 then struct 0-7 within each half
ax.set_xticks([10, 24, 38, 52])
ax.set_xticklabels(["seq (0–19)", "struct (0–7)", "seq (0–19)", "struct (0–7)"], fontsize=8)
ax.set_xlabel("Filter index  [order: incl_seq | incl_struct | skip_seq | skip_struct]", fontsize=8)
ax.set_ylabel("Bottleneck weight", fontsize=9)
ax.set_title("Variance bottleneck weights — Linear(56→1)", fontsize=10)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2166ac", label="Positive (↑ uncertainty)"),
                   Patch(color="#d6604d", label="Negative (↓ uncertainty)")],
          fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "bottleneck_weights.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved bottleneck_weights.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — 2D heatmap: MFE × predicted PSI → mean variance
# ══════════════════════════════════════════════════════════════════════════════

N_H = 10

def make_even_bins(values, n):
    edges = np.linspace(values.min(), values.max(), n + 1)
    edges[0] -= 1e-10; edges[-1] += 1e-10
    labels = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return np.asarray(labels).astype(int), edges

xi, x_edges = make_even_bins(mfe,  N_H)
yi, y_edges = make_even_bins(pred, N_H)
nx, ny = len(x_edges) - 1, len(y_edges) - 1

grid   = np.full((nx, ny), np.nan)
counts = np.zeros((nx, ny), dtype=int)
for i in range(nx):
    for j in range(ny):
        mask = (xi == i) & (yi == j)
        counts[i, j] = mask.sum()
        if mask.sum() > 0:
            grid[i, j] = var[mask].mean()

x_mids = 0.5 * (x_edges[:-1] + x_edges[1:])
y_mids = 0.5 * (y_edges[:-1] + y_edges[1:])

fig, ax = plt.subplots(figsize=(8, 7))
m = ax.pcolormesh(x_edges, y_edges, grid.T, cmap="Blues",
                  vmin=np.nanmin(grid), vmax=np.nanmax(grid))
plt.colorbar(m, ax=ax, label="Mean predicted variance (logit space)")
for i in range(nx):
    for j in range(ny):
        if counts[i, j] > 0:
            xc = 0.5 * (x_edges[i] + x_edges[i + 1])
            yc = 0.5 * (y_edges[j] + y_edges[j + 1])
            ax.text(xc, yc, f"n={counts[i,j]:,}", ha="center", va="center",
                    fontsize=5, color="black", clip_on=True)
ax.set_xticks(x_mids)
ax.set_xticklabels([f"{v:.1f}" for v in x_mids], fontsize=6.5, rotation=45, ha="right")
ax.set_yticks(y_mids)
ax.set_yticklabels([f"{v:.3f}" for v in y_mids], fontsize=6.5)
ax.set_xlabel("MFE (kcal/mol)", fontsize=9)
ax.set_ylabel("Predicted PSI", fontsize=9)
ax.set_title("Mean predicted variance by MFE × Predicted PSI", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "heatmap_variance_mfe_psi_flank_150_30.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved heatmap_variance_mfe_psi_flank_150_30.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3a — Variance vs binned MFE
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 4))
mids, means, _ = bin_stats(mfe, var)
ax.plot(mids, means, color=BLUES(0.7), linewidth=1.5, marker="o", markersize=4)
ax.set_xlabel("MFE (kcal/mol)", fontsize=9)
ax.set_ylabel("Mean predicted variance", fontsize=9)
ax.set_title("Predicted Variance vs MFE", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "var_vs_mfe_flank_150_30.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved var_vs_mfe_flank_150_30.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3b — Variance vs structure score (vienna60 structure as proxy)
# ══════════════════════════════════════════════════════════════════════════════

var_struct = var[struct_df.index]
fig, ax = plt.subplots(figsize=(7, 4))
mids2, means2, _ = bin_stats(struct_scores, var_struct)
ax.plot(mids2, means2, color=BLUES(0.7), linewidth=1.5, marker="o", markersize=4)
ax.set_xlabel("Weighted structure score  (GC=3, AU=2, GU=1 per base pair)", fontsize=9)
ax.set_ylabel("Mean predicted variance", fontsize=9)
ax.set_title("Predicted Variance vs Structure Score", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "var_vs_structure_flank_150_30.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved var_vs_structure_flank_150_30.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Calibration: mean |residual| vs binned predicted variance
# ══════════════════════════════════════════════════════════════════════════════

# bin residual into equal-width bins, show mean variance per bin
def bin_stats_equal_width(x_vals, y_vals, n=N_BINS):
    edges = np.linspace(x_vals.min(), x_vals.max(), n + 1)
    edges[0] -= 1e-10; edges[-1] += 1e-10
    labels = pd.cut(x_vals, bins=edges, labels=False, include_lowest=True)
    mids, means = [], []
    for b in range(n):
        sel = labels == b
        if sel.sum() < 2: continue
        mids.append(x_vals[sel].mean())
        means.append(y_vals[sel].mean())
    return np.array(mids), np.array(means)

mids4, means4 = bin_stats_equal_width(resid, var)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mids4, means4, color=BLUES(0.7), linewidth=1.5, marker="o", markersize=4)
ax.set_xlabel("Mean |predicted PSI − true PSI| (binned)", fontsize=9)
ax.set_ylabel("Mean predicted variance (logit space)", fontsize=9)
ax.set_title("Predicted variance against binned mean residual", fontsize=10)
ax.text(0.03, 0.93, f"n={len(df):,}", transform=ax.transAxes, fontsize=8, va="top")
fig.tight_layout()
fig.savefig(OUT / "calibration_var_vs_residual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved calibration_var_vs_residual.png")
