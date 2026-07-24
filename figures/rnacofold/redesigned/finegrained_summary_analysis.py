"""
Fine-grained RNAduplex analysis.

Chunks:
  Upstream nearest 60nt (3 × 20nt, pos in LEFT_FLANK_130, 130 = nearest to exon):
    up_1_20   left:110:130  (1–20nt from exon)
    up_21_40  left:90:110   (21–40nt from exon)
    up_41_60  left:70:90    (41–60nt from exon)

  Downstream first 100nt (5 × 20nt):
    ds_1_20   right:0:20
    ds_21_40  right:20:40
    ds_41_60  right:40:60
    ds_61_80  right:60:80
    ds_81_100 right:80:100

Comparison: bottom 2% MFE vs top 90% MFE (exclude middle 8%).
Shuffles: 100 seeds (42–141) from test_rnaduplex_finegrained_multiseed_100.csv.
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

CHUNKS = ["up_1_20", "up_21_40", "up_41_60",
          "ds_1_20", "ds_21_40", "ds_41_60", "ds_61_80", "ds_81_100"]

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


ann     = pd.read_csv(BASE / "data/test_annotated.csv")
real_df = prep(ann, pd.read_csv(BASE / "data/test_rnaduplex_finegrained.csv"))

# ── Real p-values ──────────────────────────────────────────────────────────────

real_rows = []
print(f"{'chunk':<15} {'p_mean':>12} {'p_levene':>12}")
for chunk in CHUNKS:
    pm, pl = pvals(real_df, f"mfe_{chunk}")
    real_rows.append({"chunk": chunk, "p_mean": pm, "p_levene": pl})
    print(f"{chunk:<15} {pm:>12.3e} {pl:>12.3e}")

# ── Shuffle p-values ───────────────────────────────────────────────────────────

ms_path = BASE / "data/test_rnaduplex_finegrained_multiseed_100.csv"
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
    print(f"\nMultiseed CSV not found: {ms_path}")
    print("Submit rnaduplex_finegrained_multiseed.sbatch on HPC first.")

# ── Save p-value table ─────────────────────────────────────────────────────────

table_rows = []
for r in real_rows:
    table_rows.append({"chunk": r["chunk"], "type": "real", "seed": "-",
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})
for r in shuffle_rows:
    table_rows.append({"chunk": r["chunk"], "type": "shuffle", "seed": r["seed"],
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})

table = pd.DataFrame(table_rows)
table_path = OUT / "finegrained_pvalue_table.csv"
table.to_csv(table_path, index=False)
print(f"\nSaved p-value table: {table_path.name}")

# ── Summary figure ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
x = np.arange(len(CHUNKS))
tick_labels = [c.replace("_", "\n") for c in CHUNKS]
sep_x = 2.5

if shuffle_rows:
    ms_table = pd.DataFrame(shuffle_rows)
    for ci, chunk in enumerate(CHUNKS):
        ps = ms_table[ms_table["chunk"] == chunk]["p_mean"].values
        ax.scatter(np.full(len(ps), ci), -np.log10(ps),
                   color="#aaaaaa", s=20, alpha=0.5, zorder=3,
                   label=f"Shuffled (seeds {SEED_START}–{SEED_START+N_SEEDS-1})" if ci == 0 else "")

real_ps = [-np.log10(r["p_mean"]) for r in real_rows]
ax.scatter(x, real_ps, color="#d6604d", s=80, zorder=5, marker="*", label="Real")

ax.axhline(-np.log10(0.05), color="black", linewidth=0.8,
           linestyle=":", alpha=0.6, label="p=0.05")
ax.axvline(sep_x, color="#888888", linewidth=1.0, linestyle="--", alpha=0.5)
ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 4
ax.text(sep_x - 1.5, ymax, "upstream",   ha="center", va="top", fontsize=9, color="#555555")
ax.text(sep_x + 2.0, ymax, "downstream", ha="center", va="top", fontsize=9, color="#555555")
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_ylabel("-log10(p)", fontsize=10)
ax.legend(fontsize=8)
fig.suptitle("RNA duplex MFE significance - Finer partitions", fontsize=12)
fig.tight_layout()
p_out = OUT / "finegrained_summary_pvals.png"
fig.savefig(p_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p_out.name}")

# ── Empirical significance (seeds with higher p than real) ────────────────────

if shuffle_rows:
    ms_table = pd.DataFrame(shuffle_rows)
    print(f"\nEmpirical significance (seeds with p_mean > real — higher = less significant than real):")
    print(f"{'chunk':<15} {'n_higher':>10}  {'empirical_p':>12}")
    for r in real_rows:
        chunk = r["chunk"]
        seed_ps = ms_table[ms_table["chunk"] == chunk]["p_mean"].values
        n_higher = (seed_ps > r["p_mean"]).sum()
        print(f"{chunk:<15} {n_higher:>10}  {(N_SEEDS - n_higher) / N_SEEDS:>12.2f}")
