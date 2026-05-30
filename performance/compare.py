"""Cross-config comparison: overlaid KL plots, summary table, feature density histograms.

Usage:
    python performance/compare.py \\
      --csv data/test_annotated.csv \\
      --configs baseline vienna60 flank_40_30 \\
      --out-dir performance/generated_files/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

matplotlib.use("Agg")

from performance.lib import plots as P

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-config comparison plots and summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",     type=Path, default=BASE_DIR / "data" / "test_annotated.csv")
    p.add_argument("--configs", nargs="+", required=True,
                   help="Config names to compare, e.g. baseline vienna60 flank_40_30")
    p.add_argument("--out-dir", type=Path,
                   default=BASE_DIR / "performance" / "generated_files" / "comparison")
    p.add_argument("--bins",    type=int, default=20)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Comparison plots ──────────────────────────────────────────────────────────

def kl_distributions(df: pd.DataFrame, configs: list[str], out_dir: Path) -> None:
    """Overlaid log10(KL) histograms for all configs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg, color in zip(configs, _COLORS):
        kl_col = f"kl_{cfg}"
        if kl_col not in df.columns:
            print(f"  Warning: {kl_col} not found, skipping")
            continue
        kl = df[kl_col].dropna().values
        ax.hist(np.log10(kl + 1e-4), bins=80, alpha=0.5, color=color,
                label=f"{cfg} (n={len(kl):,})", density=True)

    ax.set_xlabel("log10(KL + 1e-4)")
    ax.set_ylabel("Density")
    ax.set_title("KL distribution by config")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "kl_distributions.png")


def kl_vs_mfe(df: pd.DataFrame, configs: list[str], out_dir: Path, bins: int) -> None:
    """Mean KL vs MFE bin — all configs on same axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg, color in zip(configs, _COLORS):
        kl_col  = f"kl_{cfg}"
        mfe_col = f"MFE_{cfg}"
        if kl_col not in df.columns or mfe_col not in df.columns:
            continue
        sub = df[[mfe_col, kl_col]].dropna()
        try:
            midpoints, means = P.bin_means(sub, mfe_col, kl_col, bins)
            ax.plot(midpoints, means, marker="o", linewidth=1.5, markersize=4,
                    color=color, label=cfg)
        except Exception:
            pass

    ax.set_xlabel("MFE (kcal/mol)")
    ax.set_ylabel("Mean KL")
    ax.set_title("Mean KL vs MFE bin — all configs")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "kl_vs_MFE.png")


def density_mfe_comparison(df: pd.DataFrame, configs: list[str], out_dir: Path) -> None:
    """Density histogram: MFE of each config vs baseline MFE."""
    baseline_mfe = "MFE_baseline"
    if baseline_mfe not in df.columns:
        print("  Skipping density_MFE_comparison (MFE_baseline not in CSV)")
        return

    other_cfgs = [c for c in configs if c != "baseline" and f"MFE_{c}" in df.columns]
    if not other_cfgs:
        return

    n_plots = len(other_cfgs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, cfg in zip(axes, other_cfgs):
        mfe_col = f"MFE_{cfg}"
        sub = df[[baseline_mfe, mfe_col]].dropna()
        P.density_histogram(
            ax, sub[baseline_mfe].values, sub[mfe_col].values,
            xlabel="MFE_baseline (kcal/mol)",
            ylabel=f"MFE_{cfg} (kcal/mol)",
            title=f"MFE: baseline vs {cfg}",
        )

    fig.tight_layout()
    _save(fig, out_dir, "density_MFE_comparison.png")


def density_psi_vs_pred(df: pd.DataFrame, configs: list[str], out_dir: Path) -> None:
    """Density histogram: true PSI vs predicted PSI, one panel per config."""
    valid = [c for c in configs if f"predicted_PSI_{c}" in df.columns]
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 5))
    if len(valid) == 1:
        axes = [axes]

    for ax, cfg in zip(axes, valid):
        sub = df[["PSI", f"predicted_PSI_{cfg}"]].dropna()
        P.density_histogram(
            ax, sub["PSI"].values, sub[f"predicted_PSI_{cfg}"].values,
            xlabel="PSI (true)", ylabel=f"PSI predicted ({cfg})",
            title=f"PSI correlation | {cfg}",
        )

    fig.tight_layout()
    _save(fig, out_dir, "density_PSI_vs_pred.png")


def stats_table(df: pd.DataFrame, configs: list[str], out_dir: Path) -> None:
    """Summary stats table: mean/median/p90/p95 KL for each config."""
    records = []
    for cfg in configs:
        kl_col = f"kl_{cfg}"
        if kl_col not in df.columns:
            continue
        kl = df[kl_col].dropna().values
        records.append({
            "config":  cfg,
            "n":       len(kl),
            "mean_kl": kl.mean(),
            "median_kl": np.median(kl),
            "p90_kl":  np.percentile(kl, 90),
            "p95_kl":  np.percentile(kl, 95),
        })
    stats_df = pd.DataFrame(records)
    stats_path = out_dir / "stats_table.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved {stats_path.name}")
    print(stats_df.to_string(index=False))
    return stats_df


# ── Summary markdown ──────────────────────────────────────────────────────────

def write_summary(out_dir: Path, configs: list[str], stats_df: pd.DataFrame) -> None:
    lines = [
        "# Cross-config comparison\n\n",
        f"**Configs compared:** {', '.join(configs)}\n\n",
        "## KL summary\n\n",
        stats_df.to_markdown(index=False) if hasattr(stats_df, "to_markdown") else str(stats_df),
        "\n\n## Plots\n",
    ]
    for png in sorted(out_dir.glob("*.png")):
        lines.append(f"\n![{png.stem}]({png.name})\n")
    (out_dir / "summary.md").write_text("".join(lines))
    print("  Saved summary.md")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv} ({len(df):,} rows)")
    print(f"Configs: {args.configs}")
    print(f"Output:  {args.out_dir}\n")

    kl_distributions(df, args.configs, args.out_dir)
    kl_vs_mfe(df, args.configs, args.out_dir, args.bins)
    density_mfe_comparison(df, args.configs, args.out_dir)
    density_psi_vs_pred(df, args.configs, args.out_dir)
    sdf = stats_table(df, args.configs, args.out_dir)
    write_summary(args.out_dir, args.configs, sdf)

    print(f"\nDone — outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
