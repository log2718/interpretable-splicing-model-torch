"""
Summary analysis for redesigned RNAduplex chunks.

1. Computes real p-values for all 8 chunks.
2. Loads multiseed shuffle CSV (seeds 42-51) and computes p-values per seed.
3. Saves a p-value table (CSV).
4. Generates a summary figure: real p-value (star) + 10 shuffled p-values (dots)
   for all 8 chunks in one plot.

Run after both CSVs are available:
  data/test_rnaduplex_redesigned.csv
  data/test_rnaduplex_redesigned_multiseed.csv
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
    thr  = df[col].quantile(0.02)
    bot  = df[df[col] <= thr]["signed_residual"].values
    rest = df[df[col] >  thr]["signed_residual"].values
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

real_table = pd.DataFrame(real_rows)

# ── Shuffle p-values (only if multiseed CSV exists) ───────────────────────────

ms_path = BASE / "data/test_rnaduplex_redesigned_multiseed_100.csv"
shuffle_rows = []

if ms_path.exists():
    ms_df = prep(ann, pd.read_csv(ms_path))
    print(f"\nMultiseed shuffle p-values (seeds {SEED_START}–{SEED_START+N_SEEDS-1}):")
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
    print(f"\nMultiseed CSV not found at {ms_path} — skipping shuffle analysis.")
    print("Run sbatch rnaduplex_redesigned_multiseed.sbatch first.")

# ── Save p-value table ─────────────────────────────────────────────────────────

table_rows = []
for r in real_rows:
    table_rows.append({"chunk": r["chunk"], "type": "real", "seed": "-",
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})
for r in shuffle_rows:
    table_rows.append({"chunk": r["chunk"], "type": "shuffle", "seed": r["seed"],
                        "p_mean": r["p_mean"], "p_levene": r["p_levene"]})

table = pd.DataFrame(table_rows)
table_path = OUT / "redesigned_pvalue_table.csv"
table.to_csv(table_path, index=False)
print(f"\nSaved p-value table: {table_path.name}")

# ── Summary figure ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
x = np.arange(len(CHUNKS))
labels = [c.replace("_", "\n") for c in CHUNKS]

for ax_i, (stat, title) in enumerate([("p_mean",   "Mean residual PSI (t-test)"),
                                        ("p_levene", "Variance of residual PSI (Levene)")]):
    ax = axes[ax_i]

    # Shuffle dots
    if shuffle_rows:
        ms_table = pd.DataFrame(shuffle_rows)
        for ci, chunk in enumerate(CHUNKS):
            ps = ms_table[ms_table["chunk"] == chunk][stat].values
            ax.scatter(np.full(len(ps), ci), -np.log10(ps),
                       color="#aaaaaa", s=30, alpha=0.7, zorder=3,
                       label=f"Shuffled (seeds {SEED_START}–{SEED_START+N_SEEDS-1})" if ci == 0 else "")

    # Real p-values
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
             f"red star = real sequence; grey dots = {N_SEEDS} mononucleotide shuffles (seeds {SEED_START}–{SEED_START+N_SEEDS-1})",
             fontsize=10, y=1.02)
fig.tight_layout()
p_out = OUT / "redesigned_summary_pvals.png"
fig.savefig(p_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p_out.name}")

# ── P-value table (CLI + PNG) ──────────────────────────────────────────────────

ms_table_df = pd.DataFrame(shuffle_rows) if shuffle_rows else None

row_labels = ["real"] + [f"seed {SEED_START + i}" for i in range(N_SEEDS)]

for stat, stat_label in [("p_mean",   "t-test (mean residual)"),
                          ("p_levene", "Levene (variance)")]:
    # Build 11 × 8 matrix
    real_row = [r[stat] for r in real_rows]
    shuf_rows = []
    if ms_table_df is not None:
        for i in range(N_SEEDS):
            seed = SEED_START + i
            shuf_rows.append([
                ms_table_df[(ms_table_df["chunk"] == c) &
                            (ms_table_df["seed"]  == seed)][stat].values[0]
                for c in CHUNKS
            ])
    mat = np.array([real_row] + shuf_rows)   # (11, 8)

    # ── CLI print ──────────────────────────────────────────────────────────────
    SIG = "\033[1;31m"   # bold red
    RST = "\033[0m"
    col_w = 13
    print(f"\n{stat_label}")
    print(f"{'':>10}" + "".join(f"{c:>{col_w}}" for c in CHUNKS))
    for ri, rl in enumerate(row_labels):
        cells = ""
        for v in mat[ri]:
            s = f"{v:.2e}"
            cells += (SIG + f"{s:>{col_w}}" + RST) if v < 0.05 else f"{s:>{col_w}}"
        print(f"{rl:>10}{cells}")

    if ms_table_df is None:
        continue

    # ── PNG table ──────────────────────────────────────────────────────────────
    cell_text   = [[f"{v:.2e}" for v in row] for row in mat]
    cell_colors = []
    for ri, row in enumerate(mat):
        row_colors = []
        for v in row:
            if ri == 0:
                row_colors.append("#ea9999" if v < 0.05 else "white")  # real row
            else:
                row_colors.append("#ea9999" if v < 0.05 else "white")  # shuffle rows
        cell_colors.append(row_colors)

    col_labels = [c.replace("_", "\n") for c in CHUNKS]
    fig_t, ax_t = plt.subplots(figsize=(18 / 2.54, 8 / 2.54))
    ax_t.axis("off")
    tbl = ax_t.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.auto_set_column_width(list(range(-1, len(CHUNKS))))
    tbl.scale(1.0, 1.3)
    for j in range(len(CHUNKS)):               # bold column headers
        tbl[0, j].set_text_props(fontweight="bold")
    for i in range(len(row_labels)):           # bold row labels
        tbl[i + 1, -1].set_text_props(fontweight="bold")
    for j in range(len(CHUNKS)):               # bold real row cells
        tbl[1, j].set_text_props(fontweight="bold")
    tbl[1, -1].set_text_props(fontweight="bold")  # bold real row label
    ax_t.set_title(
        f"RNAduplex p-values — {stat_label} (red = significant)",
        fontsize=7, pad=8,
    )
    fig_t.tight_layout(pad=0.3)
    p_tbl = OUT / f"redesigned_pvalue_table_{stat}.png"
    fig_t.savefig(p_tbl, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p_tbl.name}")
