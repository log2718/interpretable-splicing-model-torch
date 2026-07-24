"""
Summary analysis — 20 shuffle seeds, bottom 2% vs top 90% comparison.

Comparison: exons in the bottom 2% of MFE (strongest binders) vs exons
above the 10th percentile (weakest 90% by binding strength). The middle
8% (2nd–10th percentile) is excluded from both groups.

Uses first 20 seeds (42–61) from the 100-seed multiseed CSV.
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

CHUNKS = ["ss3", "up_near", "up_mid", "up_far", "up_total",
          "upstream_exon",
          "downstream_short", "downstream_long", "downstream_exon"]

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
    thr_bot = df[col].quantile(0.02)   # 2nd percentile — strongest binders
    thr_top = df[col].quantile(0.10)   # 10th percentile — cut-off for weak binders
    bot  = df[df[col] <= thr_bot]["signed_residual"].values  # bottom 2%
    rest = df[df[col] >  thr_top]["signed_residual"].values  # top 90% (weakest)
    _, pm = ttest_ind(bot, rest)
    _, pl = levene_test(bot, rest)
    return pm, pl


ann     = pd.read_csv(BASE / "data/test_annotated.csv")
real_df = prep(ann, pd.read_csv(BASE / "data/test_rnaduplex_redesigned.csv"))

# ── Real p-values ──────────────────────────────────────────────────────────────

real_rows = []
print(f"{'chunk':<20} {'p_mean':>12} {'p_levene':>12}")
for chunk in CHUNKS:
    pm, pl = pvals(real_df, f"mfe_{chunk}")
    real_rows.append({"chunk": chunk, "p_mean": pm, "p_levene": pl})
    print(f"{chunk:<20} {pm:>12.3e} {pl:>12.3e}")

# ── Shuffle p-values ───────────────────────────────────────────────────────────

ms_path = BASE / "data/test_rnaduplex_redesigned_multiseed_100.csv"
shuffle_rows = []

if ms_path.exists():
    ms_df = prep(ann, pd.read_csv(ms_path))
    print(f"\nShuffle p-values (seeds {SEED_START}–{SEED_START+N_SEEDS-1}):")
    print(f"{'chunk':<20} {'seed':>6} {'p_mean':>12} {'p_levene':>12}")
    for chunk in CHUNKS:
        for i in range(N_SEEDS):
            seed = SEED_START + i
            col  = f"mfe_{chunk}_seed_{i}"
            pm, pl = pvals(ms_df, col)
            shuffle_rows.append({"chunk": chunk, "seed": seed,
                                  "p_mean": pm, "p_levene": pl})
            print(f"{chunk:<20} {seed:>6} {pm:>12.3e} {pl:>12.3e}")
else:
    print(f"\nMultiseed CSV not found: {ms_path}")

# ── Save p-value table ─────────────────────────────────────────────────────────

table_rows = []
for r in real_rows:
    table_rows.append({"chunk": r["chunk"], "type": "real", "seed": "-",
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})
for r in shuffle_rows:
    table_rows.append({"chunk": r["chunk"], "type": "shuffle", "seed": r["seed"],
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})

table = pd.DataFrame(table_rows)
table_path = OUT / "redesigned_pvalue_table_20seed_top90.csv"
table.to_csv(table_path, index=False)
print(f"\nSaved p-value table: {table_path.name}")

# ── Summary figure ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
x = np.arange(len(CHUNKS))
labels = [c.replace("_", "\n") for c in CHUNKS]

for ax_i, (stat, title) in enumerate([("p_mean",   "Mean residual PSI (t-test)"),
                                        ("p_levene", "Variance of residual PSI (Levene)")]):
    ax = axes[ax_i]

    if shuffle_rows:
        ms_table = pd.DataFrame(shuffle_rows)
        for ci, chunk in enumerate(CHUNKS):
            ps = ms_table[ms_table["chunk"] == chunk][stat].values
            ax.scatter(np.full(len(ps), ci), -np.log10(ps),
                       color="#aaaaaa", s=30, alpha=0.7, zorder=3,
                       label=f"Shuffled (seeds {SEED_START}–{SEED_START+N_SEEDS-1})" if ci == 0 else "")

    real_ps = [-np.log10(r[stat]) for r in real_rows]
    ax.scatter(x, real_ps, color="#d6604d", s=80, zorder=5,
               marker="*", label="Real")

    ax.axhline(-np.log10(0.05), color="black", linewidth=0.8,
               linestyle=":", alpha=0.6, label="p=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("-log10(p)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle("RNAduplex shuffle vs original sequence residuals\n"
             f"bottom 2% vs top 90% by MFE | {N_SEEDS} shuffles (seeds {SEED_START}–{SEED_START+N_SEEDS-1})",
             fontsize=10, y=1.02)
fig.tight_layout()
p_out = OUT / "redesigned_summary_pvals_20seed_top90.png"
fig.savefig(p_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p_out.name}")
