"""
Multi-seed shuffle significance analysis.

For each chunk, plots the distribution of p-values across N shuffled seeds
vs the real p-value. If the real p-value is not clearly lower than the
shuffle distribution, the effect is likely a GC content artifact.

Usage:
  python figures/rnacofold/shuffle_multiseed_analysis.py \\
      --multiseed-csv data/test_rnaduplex_upstream_exon_multiseed.csv \\
      --real-csv data/test_rnaduplex_upstream_exon.csv \\
      --chunks upstream_exon \\
      --out-prefix rnaduplex_upstream_exon_multiseed
"""

from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene as levene_test

BASE = Path(__file__).resolve().parent.parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def compute_pvals(df: pd.DataFrame, mfe_col: str) -> tuple[float, float]:
    thr  = df[mfe_col].quantile(0.02)
    bot  = df[df[mfe_col] <= thr]["signed_residual"].values
    rest = df[df[mfe_col] >  thr]["signed_residual"].values
    _, p_mean   = ttest_ind(bot, rest)
    _, p_levene = levene_test(bot, rest)
    return p_mean, p_levene


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed-csv", type=Path, required=True)
    p.add_argument("--real-csv",      type=Path, required=True)
    p.add_argument("--chunks",        nargs="+", required=True,
                   help="Chunk names (without seed suffix)")
    p.add_argument("--out-prefix",    type=str,  required=True)
    return p.parse_args()


def main():
    args = parse_args()

    ann      = pd.read_csv(BASE / "data/test_annotated.csv")
    ms_df    = pd.read_csv(args.multiseed_csv)
    real_df  = pd.read_csv(args.real_csv)

    def prep(df):
        merged = ann.merge(df, on="exon", how="inner")
        merged = merged[
            merged["PSI"].notna() &
            merged["predicted_PSI_flank_150_30_unc"].notna()
        ].copy()
        merged["signed_residual"] = merged["predicted_PSI_flank_150_30_unc"] - merged["PSI"]
        return merged

    ms   = prep(ms_df)
    real = prep(real_df)

    # Detect number of seeds from column names
    n_seeds = max(
        int(c.split("_seed_")[1])
        for c in ms.columns if "_seed_" in c
    ) + 1

    print(f"Chunks: {args.chunks}  |  Seeds: {n_seeds}")

    n_chunks = len(args.chunks)
    fig, axes = plt.subplots(n_chunks, 2, figsize=(10, 4 * n_chunks))
    if n_chunks == 1:
        axes = [axes]

    for row_i, chunk in enumerate(args.chunks):
        # Real p-values
        real_col = f"mfe_{chunk}"
        p_mean_real, p_lev_real = compute_pvals(real, real_col)

        # Shuffle p-values across seeds
        p_means, p_levs = [], []
        for s in range(n_seeds):
            col = f"mfe_{chunk}_seed_{s}"
            pm, pl = compute_pvals(ms, col)
            p_means.append(pm)
            p_levs.append(pl)

        print(f"\n{chunk}")
        print(f"  Real:    p_mean={p_mean_real:.3e}  p_levene={p_lev_real:.3e}")
        for s in range(n_seeds):
            print(f"  Seed {s}:  p_mean={p_means[s]:.3e}  p_levene={p_levs[s]:.3e}")

        # Plot mean p-values
        ax = axes[row_i][0]
        ax.scatter(range(n_seeds), [-np.log10(p) for p in p_means],
                   color="#888888", s=60, zorder=3, label="Shuffled seeds")
        ax.axhline(-np.log10(p_mean_real), color="#d6604d", linewidth=2,
                   linestyle="--", label=f"Real (p={p_mean_real:.2e})")
        ax.axhline(-np.log10(0.05), color="black", linewidth=0.8,
                   linestyle=":", alpha=0.5, label="p=0.05")
        ax.set_xticks(range(n_seeds))
        ax.set_xlabel("Shuffle seed", fontsize=8)
        ax.set_ylabel("-log10(p)", fontsize=8)
        ax.set_title(f"{chunk} — mean residual p-values\nacross {n_seeds} shuffles vs real",
                     fontsize=9)
        ax.legend(fontsize=7)

        # Plot Levene p-values
        ax = axes[row_i][1]
        ax.scatter(range(n_seeds), [-np.log10(p) for p in p_levs],
                   color="#888888", s=60, zorder=3, label="Shuffled seeds")
        ax.axhline(-np.log10(p_lev_real), color="#4393c3", linewidth=2,
                   linestyle="--", label=f"Real (p={p_lev_real:.2e})")
        ax.axhline(-np.log10(0.05), color="black", linewidth=0.8,
                   linestyle=":", alpha=0.5, label="p=0.05")
        ax.set_xticks(range(n_seeds))
        ax.set_xlabel("Shuffle seed", fontsize=8)
        ax.set_ylabel("-log10(p)", fontsize=8)
        ax.set_title(f"{chunk} — Levene p-values\nacross {n_seeds} shuffles vs real",
                     fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"Shuffle control: real vs {n_seeds} mononucleotide shuffles\n"
                 f"(red/blue dashed = real; grey dots = shuffled seeds)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    p_out = OUT / f"{args.out_prefix}_pval_comparison.png"
    fig.savefig(p_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {p_out.name}")


if __name__ == "__main__":
    main()
