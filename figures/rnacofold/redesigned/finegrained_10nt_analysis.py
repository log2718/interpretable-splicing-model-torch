"""
Fine-grained 10-nt sub-region RNAduplex significance analysis.

Splits the two significant 20-nt windows (up_1_20 and ds_1_20) each into
two 10-nt halves and tests whether the effect is driven by the proximal
or distal half relative to the splice site.

Chunks (relative to LEFT_FLANK_130 / RIGHT_FLANK_100):
  up_1_10  : left:120:130  — nt  1-10 upstream of 3'SS  (most proximal)
  up_11_20 : left:110:120  — nt 11-20 upstream of 3'SS
  ds_1_10  : right:0:10   — nt  1-10 downstream of 5'SS (most proximal)
  ds_11_20 : right:10:20  — nt 11-20 downstream of 5'SS

Real data:   data/test_rnaduplex_finegrained_10nt.csv
Shuffled:    data/test_rnaduplex_finegrained_10nt_multiseed_100.csv  (100 seeds, 42-141)

Usage:
  python figures/rnacofold/redesigned/finegrained_10nt_analysis.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene as levene_test

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

CHUNKS     = ["up_1_10", "up_11_20", "ds_1_10", "ds_11_20"]
N_SEEDS    = 100
SEED_START = 42


def prep(ann, df):
    merged = ann.merge(df, on="exon", how="inner")
    merged = merged[
        merged["PSI"].notna() &
        merged["predicted_PSI_flank_150_30_unc"].notna()
    ].copy()
    merged["signed_residual"] = (merged["predicted_PSI_flank_150_30_unc"]
                                 - merged["PSI"])
    return merged


def pvals(df, col):
    thr_bot = df[col].quantile(0.02)
    thr_top = df[col].quantile(0.10)
    bot  = df[df[col] <= thr_bot]["signed_residual"].values
    rest = df[df[col] >  thr_top]["signed_residual"].values
    _, pm = ttest_ind(bot, rest)
    _, pl = levene_test(bot, rest)
    return pm, pl


ann = pd.read_csv(BASE / "data/test_annotated.csv")

real_path = BASE / "data/test_rnaduplex_finegrained_10nt.csv"
if not real_path.exists():
    raise FileNotFoundError(
        f"{real_path.name} not found — run rnaduplex_10nt_real.sbatch on HPC first."
    )
real_df = prep(ann, pd.read_csv(real_path))

# ── Real p-values ──────────────────────────────────────────────────────────────

real_rows = []
print(f"{'chunk':<15} {'p_mean':>12} {'p_levene':>12}")
for chunk in CHUNKS:
    pm, pl = pvals(real_df, f"mfe_{chunk}")
    real_rows.append({"chunk": chunk, "p_mean": pm, "p_levene": pl})
    print(f"{chunk:<15} {pm:>12.3e} {pl:>12.3e}")

# ── Shuffle p-values ───────────────────────────────────────────────────────────

ms_path = BASE / "data/test_rnaduplex_finegrained_10nt_multiseed_100.csv"
shuffle_rows = []

if ms_path.exists():
    ms_df = prep(ann, pd.read_csv(ms_path))
    print(f"\nShuffle p-values (seeds {SEED_START}–{SEED_START+N_SEEDS-1}):")
    print(f"{'chunk':<15} {'seed':>6} {'p_mean':>12} {'p_levene':>12}")
    for chunk in CHUNKS:
        for i in range(N_SEEDS):
            seed = SEED_START + i
            col  = f"mfe_{chunk}_seed_{i}"
            pm, pl = pvals(ms_df, col)
            shuffle_rows.append({"chunk": chunk, "seed": seed,
                                  "p_mean": pm, "p_levene": pl})
            print(f"{chunk:<15} {seed:>6} {pm:>12.3e} {pl:>12.3e}")
else:
    print(f"\nMultiseed CSV not found: {ms_path.name}")
    print("Submit rnaduplex_10nt_multiseed.sbatch on HPC first.")

# ── Save p-value table ─────────────────────────────────────────────────────────

table_rows = []
for r in real_rows:
    table_rows.append({"chunk": r["chunk"], "type": "real", "seed": "-",
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})
for r in shuffle_rows:
    table_rows.append({"chunk": r["chunk"], "type": "shuffle", "seed": r["seed"],
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})

table = pd.DataFrame(table_rows)
table_path = OUT / "finegrained_10nt_pvalue_table.csv"
table.to_csv(table_path, index=False)
print(f"\nSaved p-value table: {table_path.name}")

# ── Summary figure ─────────────────────────────────────────────────────────────

chunk_labels = {
    "up_1_10":  "up 1-10\n(proximal)",
    "up_11_20": "up 11-20\n(distal)",
    "ds_1_10":  "ds 1-10\n(proximal)",
    "ds_11_20": "ds 11-20\n(distal)",
}

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
x = np.arange(len(CHUNKS))

if shuffle_rows:
    ms_table = pd.DataFrame(shuffle_rows)
    for ci, chunk in enumerate(CHUNKS):
        ps = ms_table[ms_table["chunk"] == chunk]["p_mean"].values
        ax.scatter(np.full(len(ps), ci), -np.log10(ps),
                   color="#aaaaaa", s=20, alpha=0.5, zorder=3,
                   label=f"Shuffled (seeds {SEED_START}–{SEED_START+N_SEEDS-1})" if ci == 0 else "")

real_ps = [-np.log10(r["p_mean"]) for r in real_rows]
ax.scatter(x, real_ps, color="#d6604d", s=100, zorder=5, marker="*", label="Real")

ax.axhline(-np.log10(0.05), color="black", linewidth=0.8,
           linestyle=":", alpha=0.6, label="p=0.05")
ax.axvline(1.5, color="#888888", linewidth=1.0, linestyle="--", alpha=0.5)
ymax = max(ax.get_ylim()[1], 4)
ax.set_ylim(bottom=0, top=ymax)
ax.text(0.5,  ymax * 0.97, "upstream",   ha="center", va="top", fontsize=9, color="#555555")
ax.text(2.5,  ymax * 0.97, "downstream", ha="center", va="top", fontsize=9, color="#555555")
ax.set_xticks(x)
ax.set_xticklabels([chunk_labels[c] for c in CHUNKS], fontsize=9)
ax.set_ylabel("-log10(p_mean)", fontsize=10)
ax.legend(fontsize=8)
fig.suptitle("RNAduplex MFE significance — 10-nt sub-regions of up_1_20 and ds_1_20\n"
             "(red star = real; grey = 100 mononucleotide-shuffled seeds)", fontsize=11)
fig.tight_layout()
p_out = OUT / "finegrained_10nt_pval_comparison.png"
fig.savefig(p_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p_out.name}")

# ── Empirical significance ─────────────────────────────────────────────────────

if shuffle_rows:
    ms_table = pd.DataFrame(shuffle_rows)
    print(f"\nEmpirical significance (fraction of shuffled seeds with p_mean > real):")
    print(f"{'chunk':<15} {'real_p_mean':>14} {'n_shuf_higher':>15} {'empirical_p':>13}")
    for r in real_rows:
        chunk = r["chunk"]
        seed_ps = ms_table[ms_table["chunk"] == chunk]["p_mean"].values
        n_higher = (seed_ps > r["p_mean"]).sum()
        emp_p = (N_SEEDS - n_higher) / N_SEEDS
        print(f"{chunk:<15} {r['p_mean']:>14.3e} {n_higher:>15}  {emp_p:>12.2f}")
