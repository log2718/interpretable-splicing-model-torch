"""
2D heatmap: mean predicted variance per (MFE_vienna60 × predicted_PSI_vienna60_unc) cell.

x-axis : MFE_vienna60             – 10 equal-frequency bins
y-axis : predicted_PSI_vienna60_unc – 10 equal-frequency bins
colour : mean var_vienna60_unc (logit-space variance) per cell
Each cell is annotated with the sample count n=.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).resolve().parent.parent.parent
DATA  = BASE / "data" / "test_annotated.csv"
OUT   = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA, usecols=["MFE_vienna60", "predicted_PSI_vienna60_unc", "var_vienna60_unc"])
df = df.dropna().reset_index(drop=True)
print(f"Loaded {len(df):,} examples")

x = df["MFE_vienna60"].values.astype(float)           # x-axis (MFE, kcal/mol)
y = df["predicted_PSI_vienna60_unc"].values.astype(float)  # y-axis (predicted PSI)
v = df["var_vienna60_unc"].values.astype(float)        # colour (variance, logit space)

# ── Equal-frequency binning ───────────────────────────────────────────────────

N_BINS = 10

def make_even_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_indices 0..n-1, bin_edges length n+1) via equal-width cut."""
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    edges[0]  -= 1e-10                          # ensure minimum value is included
    edges[-1] += 1e-10                          # ensure maximum value is included
    labels = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    arr = np.asarray(labels)
    return arr.astype(int), edges

xi, x_edges = make_even_bins(x, N_BINS)
yi, y_edges = make_even_bins(y, N_BINS)

nx = len(x_edges) - 1
ny = len(y_edges) - 1

# ── Build grid ────────────────────────────────────────────────────────────────

grid   = np.full((nx, ny), np.nan)
counts = np.zeros((nx, ny), dtype=int)

for i in range(nx):
    for j in range(ny):
        mask = (xi == i) & (yi == j)
        n = int(mask.sum())
        counts[i, j] = n
        if n > 0:
            grid[i, j] = v[mask].mean()

# ── Midpoints for tick labels ─────────────────────────────────────────────────

x_mids = 0.5 * (x_edges[:-1] + x_edges[1:])
y_mids = 0.5 * (y_edges[:-1] + y_edges[1:])

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 7))

vmin = float(np.nanmin(grid))
vmax = float(np.nanmax(grid))

m = ax.pcolormesh(x_edges, y_edges, grid.T, cmap="Blues", vmin=vmin, vmax=vmax)
cbar = plt.colorbar(m, ax=ax)
cbar.set_label("Mean predicted variance (logit space)", fontsize=9)

# Per-cell n= annotations
for i in range(nx):
    for j in range(ny):
        if counts[i, j] > 0:
            xc = 0.5 * (x_edges[i] + x_edges[i + 1])
            yc = 0.5 * (y_edges[j] + y_edges[j + 1])
            ax.text(xc, yc, f"n={counts[i, j]:,}",
                    ha="center", va="center", fontsize=5,
                    color="black", clip_on=True)

# Tick labels at bin midpoints
ax.set_xticks(x_mids)
ax.set_xticklabels([f"{v:.1f}" for v in x_mids], fontsize=6.5, rotation=45, ha="right")
ax.set_yticks(y_mids)
ax.set_yticklabels([f"{v:.3f}" for v in y_mids], fontsize=6.5)

ax.set_xlabel("MFE (kcal/mol) — ViennaRNA 60°C", fontsize=9)
ax.set_ylabel("Predicted PSI", fontsize=9)
ax.set_title(
    "Mean predicted variance (logit space)\nby MFE × Predicted PSI",
    fontsize=10,
)

fig.tight_layout()
out_path = OUT / "heatmap_variance_mfe_psi.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
