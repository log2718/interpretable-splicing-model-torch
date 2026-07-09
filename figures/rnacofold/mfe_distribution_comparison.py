"""
Compare RNAcofold vs RNAduplex MFE distributions on the test set.

Panel 1: Overlapping histograms of intra-exon MFE
         (RNAcofold first-half vs second-half vs RNAduplex first vs second half)
Panel 2: Scatter — RNAcofold MFE vs RNAduplex MFE per exon (intra-exon)
Panel 3: Overlapping histograms of exon-intron MFE for each chunk (RNAduplex interact)

Source files:
  data/test_rnacofold_intrinsic.csv  → mfe_intrinsic (RNAcofold), mfe_rnaduplex (RNAduplex)
  data/test_rnaduplex_interact.csv   → mfe_up_far, mfe_up_mid, mfe_up_near, mfe_down
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

intr   = pd.read_csv(BASE / "data/test_rnacofold_intrinsic.csv").dropna()
inter  = pd.read_csv(BASE / "data/test_rnaduplex_interact.csv").dropna()

# ── Fig 1: Intra-exon MFE distributions ──────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: overlapping histogram
ax = axes[0]
bins = np.linspace(
    min(intr["mfe_intrinsic"].min(), intr["mfe_rnaduplex"].min()) - 1,
    max(intr["mfe_intrinsic"].max(), intr["mfe_rnaduplex"].max()) + 1,
    60,
)
ax.hist(intr["mfe_intrinsic"], bins=bins, alpha=0.55, color="#d6604d",
        label=f"RNAcofold  (mean={intr['mfe_intrinsic'].mean():.1f})", density=True)
ax.hist(intr["mfe_rnaduplex"], bins=bins, alpha=0.55, color="#4393c3",
        label=f"RNAduplex  (mean={intr['mfe_rnaduplex'].mean():.1f})", density=True)
ax.set_xlabel("MFE (kcal/mol)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Intra-exon MFE: RNAcofold vs RNAduplex\n(first half vs second half of exon)",
             fontsize=9)
ax.legend(fontsize=8)

# Panel B: scatter
ax = axes[1]
x = intr["mfe_rnaduplex"].values
y = intr["mfe_intrinsic"].values
r, p = pearsonr(x, y)
alpha = min(0.4, 300 / len(x))
ax.scatter(x, y, s=4, alpha=alpha, color="#555555", linewidths=0)
lo = min(x.min(), y.min()) - 1
hi = max(x.max(), y.max()) + 1
ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y=x")
ax.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=9)
ax.set_ylabel("RNAcofold MFE (kcal/mol)", fontsize=9)
ax.set_title(f"RNAcofold vs RNAduplex per exon\nr={r:.3f}, p={p:.1e}", fontsize=9)
ax.legend(fontsize=8)

fig.suptitle("Intra-exon folding: RNAcofold vs RNAduplex MFE distributions", fontsize=10)
fig.tight_layout()
p1 = OUT / "mfe_distribution_intrinsic.png"
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p1.name}")
print(f"  RNAcofold  mean={intr['mfe_intrinsic'].mean():.2f}  "
      f"median={intr['mfe_intrinsic'].median():.2f}  "
      f"min={intr['mfe_intrinsic'].min():.2f}")
print(f"  RNAduplex  mean={intr['mfe_rnaduplex'].mean():.2f}  "
      f"median={intr['mfe_rnaduplex'].median():.2f}  "
      f"min={intr['mfe_rnaduplex'].min():.2f}")

# ── Fig 2: Exon-intron interact MFE distributions ────────────────────────────

CHUNKS = ["up_far", "up_mid", "up_near", "down"]
COLORS = ["#1b7837", "#762a83", "#e08214", "#2166ac"]

fig2, ax2 = plt.subplots(figsize=(8, 4))

all_vals = np.concatenate([inter[f"mfe_{c}"].values for c in CHUNKS])
bins2 = np.linspace(all_vals.min() - 1, all_vals.max() + 1, 60)

for chunk, color in zip(CHUNKS, COLORS):
    vals = inter[f"mfe_{chunk}"].values
    ax2.hist(vals, bins=bins2, alpha=0.45, color=color, density=True,
             label=f"{chunk}  (mean={vals.mean():.1f})")

ax2.set_xlabel("RNAduplex MFE (kcal/mol)", fontsize=9)
ax2.set_ylabel("Density", fontsize=9)
ax2.set_title("Exon-intron RNAduplex MFE — 4 chunks", fontsize=10)
ax2.legend(fontsize=8)
fig2.tight_layout()
p2 = OUT / "mfe_distribution_interact.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p2.name}")
for chunk in CHUNKS:
    v = inter[f"mfe_{chunk}"]
    print(f"  {chunk:<8}  mean={v.mean():.2f}  median={v.median():.2f}  "
          f"min={v.min():.2f}  bottom2%={v.quantile(0.02):.2f}")
