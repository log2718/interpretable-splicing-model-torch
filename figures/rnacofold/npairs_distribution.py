"""Visual inspection of n_pairs distributions to identify natural cutoffs."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "data/test_rnacofold_interact.csv")

CHUNKS = ["up_far", "up_mid", "up_near", "down"]
COLS   = [f"n_pairs_{c}" for c in CHUNKS]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for i, (chunk, col) in enumerate(zip(CHUNKS, COLS)):
    vals = df[col].values

    # ── Row 0: histogram ──────────────────────────────────────────────────────
    ax = axes[0, i]
    ax.hist(vals, bins=40, color="#2166ac", alpha=0.8, edgecolor="white", linewidth=0.3)
    for pct in [95, 98, 99]:
        t = np.percentile(vals, pct)
        ax.axvline(t, linestyle="--", linewidth=1,
                   label=f"p{pct}={t:.0f}")
    ax.set_title(chunk, fontsize=10)
    ax.set_xlabel("n_pairs (intermolecular)", fontsize=8)
    ax.set_ylabel("Count" if i == 0 else "", fontsize=8)
    ax.legend(fontsize=7)

    # ── Row 1: sorted values (scree) ──────────────────────────────────────────
    ax2 = axes[1, i]
    sorted_vals = np.sort(vals)[::-1]
    x = np.arange(1, len(sorted_vals) + 1)
    ax2.plot(x, sorted_vals, color="#2166ac", linewidth=0.8)
    # Mark 1%, 2%, 5% cutoffs
    for pct, color in [(1, "#d6604d"), (2, "#f4a582"), (5, "#92c5de")]:
        n_cut = int(len(vals) * pct / 100)
        y_cut = sorted_vals[n_cut - 1]
        ax2.axhline(y_cut, linestyle="--", linewidth=0.8, color=color,
                    label=f"top {pct}% (n={n_cut}, ≥{y_cut:.0f} pairs)")
        ax2.axvline(n_cut, linestyle=":", linewidth=0.6, color=color)
    ax2.set_xlim(0, len(vals) * 0.15)   # zoom to top 15% for clarity
    ax2.set_xlabel("Rank (most → least pairs)", fontsize=8)
    ax2.set_ylabel("n_pairs" if i == 0 else "", fontsize=8)
    ax2.set_title(f"{chunk} — sorted (top 15% shown)", fontsize=9)
    ax2.legend(fontsize=7)

fig.suptitle("n_pairs distribution per chunk — looking for natural cutoffs", fontsize=11)
fig.tight_layout()
out_path = OUT / "npairs_distribution.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# Also print percentile table
print("\nn_pairs percentile table:")
print(f"{'chunk':<12}", end="")
for p in [90, 93, 95, 97, 98, 99, 99.5]:
    print(f"  p{p:4.1f}", end="")
print()
for chunk, col in zip(CHUNKS, COLS):
    vals = df[col].values
    print(f"{chunk:<12}", end="")
    for p in [90, 93, 95, 97, 98, 99, 99.5]:
        print(f"  {np.percentile(vals, p):5.1f}", end="")
    print()
