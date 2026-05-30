"""Generate all analysis plots and a summary for one config.

Reads data/test_annotated.csv (or --csv), uses the {metric}_{config_name} column
naming convention, and writes PNGs + stats.csv + summary.md to the output directory.

Usage:
    python performance/analyze.py \\
      --csv data/test_annotated.csv \\
      --config-name baseline \\
      --out-dir performance/generated_files/baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from performance.lib import plots as P

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate analysis plots for one training config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",         type=Path, default=BASE_DIR / "data" / "test_annotated.csv")
    p.add_argument("--config-name", required=True, help="Config label, e.g. baseline or vienna60")
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: performance/generated_files/{config-name})",
    )
    p.add_argument("--bins", type=int, default=20, help="Quantile bins for 1D binned plots")
    return p.parse_args()


# ── Per-config analysis ───────────────────────────────────────────────────────

def run(df: pd.DataFrame, cfg: str, out_dir: Path, bins: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    kl_col   = f"kl_{cfg}"
    mfe_col  = f"MFE_{cfg}"
    freq_col = f"freq_MFE_{cfg}"
    div_col  = f"ens_div_{cfg}"

    missing = [c for c in [kl_col, mfe_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for config {cfg!r}: {missing}. Run enrich.py first.")

    # Secondary features may be NaN if not yet computed (e.g., vienna60 ensemble at 60°C).
    # Only include a secondary feature if it has at least one non-NaN value.
    secondary = []
    for col, label in [(freq_col, "Freq MFE"), (div_col, "Ensemble diversity")]:
        if col in df.columns and df[col].notna().any():
            secondary.append((col, label))
        else:
            print(f"  Skipping {col} (all NaN — run enrich.py --rnafold-only to fill)")

    df = df.dropna(subset=[kl_col, mfe_col]).copy()
    print(f"  {len(df):,} rows with kl + MFE for {cfg}")

    feature_meta = [(mfe_col, "MFE (kcal/mol)")] + secondary

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close("all")
        print(f"  Saved {path.name}")

    # ── 1. Line plots — mean KL per feature bin ───────────────────────────────
    for col, label in feature_meta:
        save(P.line_mean_kl(df, col, kl_col, bins=bins, feature_label=label),
             f"line_kl_{col}.png")

    # ── 2. Violin plots — linear KL per feature bin ───────────────────────────
    for col, label in feature_meta:
        save(P.violin_kl_by_feature(df, col, kl_col, bins=bins, log_scale=False, feature_label=label),
             f"violin_{col}.png")

    # ── 3. Violin plots — log KL per feature bin ──────────────────────────────
    for col, label in feature_meta:
        save(P.violin_kl_by_feature(df, col, kl_col, bins=bins, log_scale=True, feature_label=label),
             f"violin_{col}_log.png")

    # ── 4. 1D KL histogram ────────────────────────────────────────────────────
    save(P.hist_kl(df, kl_col), "hist_kl.png")

    # ── 5. Theoretical KL curve ───────────────────────────────────────────────
    save(P.kl_curve(), "kl_curve.png")

    # ── 6. Boxplot + jitter (3-panel) ─────────────────────────────────────────
    save(P.boxplot_kl_panels(df, kl_col), "boxplot_kl.png")

    # ── 7. Proportion P(KL > 1) per feature bin ───────────────────────────────
    for col, label in feature_meta:
        save(P.proportion_above_threshold(df, col, kl_col, bins=bins, feature_label=label),
             f"proportion_{col}.png")

    # ── 8. 3×3 heatmap (MFE × each secondary feature) ────────────────────────
    for col, label in feature_meta[1:]:
        save(P.heatmap_3x3(df, mfe_col, col, kl_col,
                           mfe_label="MFE (kcal/mol)", feature_label=label),
             f"heatmap3x3_{mfe_col}_{col}.png")

    # ── 9a. 2D KL stat heatmap (pcolormesh + LogNorm, YlOrRd) ─────────────────
    import matplotlib.pyplot as plt
    for col, label in feature_meta[1:]:
        fig, ax = plt.subplots(figsize=(6, 5))
        P.kl_heatmap_2d(ax, df, mfe_col, col, kl_col,
                        stat="median",
                        xlabel="MFE (kcal/mol)", ylabel=label,
                        title=f"Median KL | {cfg}")
        save(fig, f"heatmap2d_{mfe_col}_{col}.png")

    # ── 9b. Density histogram (pcolormesh + LogNorm, Blues) ───────────────────
    psi_col = f"predicted_PSI_{cfg}"
    if psi_col in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        P.density_histogram(ax, df["PSI"].values, df[psi_col].values,
                            xlabel="PSI (true)", ylabel=f"PSI ({cfg} predicted)",
                            title=f"PSI density | {cfg}")
        save(fig, "density_PSI_vs_pred.png")

    for col, label in feature_meta[1:]:
        fig, ax = plt.subplots(figsize=(6, 5))
        P.density_histogram(ax, df[mfe_col].values, df[col].values,
                            xlabel="MFE (kcal/mol)", ylabel=label,
                            title=f"Feature density | {cfg}")
        save(fig, f"density_{mfe_col}_{col}.png")

    # ── 10. 10×10 violin grid ─────────────────────────────────────────────────
    for col, label in feature_meta[1:]:
        save(P.violin_grid(df, mfe_col, col, kl_col, n_bins=10,
                           x_label="MFE (kcal/mol)", y_label=label),
             f"violin_grid_{mfe_col}_{col}.png")
        save(P.violin_grid(df, mfe_col, col, kl_col, n_bins=10, log_scale=True,
                           x_label="MFE (kcal/mol)", y_label=label),
             f"violin_grid_{mfe_col}_{col}_log.png")

    # ── stats.csv ─────────────────────────────────────────────────────────────
    kl = df[kl_col].values
    records = [{"config": cfg, "metric": "kl", "mean": kl.mean(), "median": np.median(kl),
                "p90": np.percentile(kl, 90), "p95": np.percentile(kl, 95), "n": len(kl)}]
    for col, label in feature_meta:
        vals = df[col].values
        records.append({"config": cfg, "metric": col, "mean": vals.mean(), "median": np.median(vals),
                        "p90": np.percentile(vals, 90), "p95": np.percentile(vals, 95), "n": len(vals)})
    stats_df = pd.DataFrame(records)
    stats_path = out_dir / "stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved {stats_path.name}")

    # ── summary.md ────────────────────────────────────────────────────────────
    lines = [
        f"# Analysis: {cfg}\n",
        f"**Exons:** {len(df):,}  \n",
        f"**KL mean:** {kl.mean():.6f}  **KL median:** {np.median(kl):.6f}  "
        f"**KL p90:** {np.percentile(kl, 90):.6f}  **KL p95:** {np.percentile(kl, 95):.6f}\n\n",
        "## Plots\n",
    ]
    for png in sorted(out_dir.glob("*.png")):
        lines.append(f"![{png.stem}]({png.name})\n\n")
    (out_dir / "summary.md").write_text("".join(lines))
    print(f"  Saved summary.md")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    out_dir = args.out_dir or (
        BASE_DIR / "performance" / "generated_files" / args.config_name
    )

    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv} ({len(df):,} rows)")
    print(f"Config: {args.config_name}")
    print(f"Output: {out_dir}")

    run(df, args.config_name, out_dir, args.bins)
    print(f"\nDone — outputs in {out_dir}/")


if __name__ == "__main__":
    main()
