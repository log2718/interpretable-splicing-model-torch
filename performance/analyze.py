"""Generate all analysis plots and a summary for one config.

Three sections are produced for every run:
  Section 1 (full data):   mean KL + p90 KL vs feature bin (20 bins)
  Section 2 (full data):   pcolormesh mean-KL heatmaps for all feature pairs (10×10 bins)
  Section 3 (restricted):  same as 1 & 2 but limited to the 20% of examples
                            with the most negative MFE (first 4 equal-freq bins)

Usage:
    python performance/analyze.py \\
      --csv data/test_annotated.csv \\
      --config-name flank_40_30 \\
      --out-dir performance/generated_files/flank_40_30
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    p.add_argument("--config-name", required=True, help="Config label, e.g. baseline or flank_40_30")
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: performance/generated_files/{config-name})",
    )
    p.add_argument("--bins", type=int, default=20, help="Quantile bins for 1D binned plots")
    return p.parse_args()


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clf()
    plt.close("all")
    print(f"  Saved {path.name}")


# ── Section 1: 1D binned line plots ──────────────────────────────────────────

def _section1(df: pd.DataFrame, feature_meta: list[tuple[str, str]], kl_col: str,
               out_dir: Path, bins: int, suffix: str = "") -> None:
    """Mean KL and p90 KL vs feature bin for every feature in feature_meta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, label in feature_meta:
        _save(P.line_mean_kl(df, col, kl_col, bins=bins, feature_label=label),
              out_dir, f"s1_mean_kl_{col}{suffix}.png")
        _save(P.line_p90_kl(df, col, kl_col, bins=bins, feature_label=label),
              out_dir, f"s1_p90_kl_{col}{suffix}.png")


# ── Section 2: 2D pcolormesh mean-KL heatmaps ────────────────────────────────

def _section2(df: pd.DataFrame, feature_meta: list[tuple[str, str]], kl_col: str,
               out_dir: Path, cfg: str, suffix: str = "") -> None:
    """pcolormesh mean-KL heatmap for every pair of continuous features."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (xcol, xlabel), (ycol, ylabel) in itertools.combinations(feature_meta, 2):
        fig, ax = plt.subplots(figsize=(6, 5))
        P.kl_heatmap_2d(
            ax, df, xcol, ycol, kl_col,
            n_bins=10, stat="mean",
            xlabel=xlabel, ylabel=ylabel,
            title=f"Mean KL | {cfg}{' (restricted)' if suffix else ''}",
        )
        _save(fig, out_dir, f"s2_heatmap_{xcol}_x_{ycol}{suffix}.png")


# ── Per-config analysis ───────────────────────────────────────────────────────

def run(df: pd.DataFrame, cfg: str, out_dir: Path, bins: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    kl_col   = f"kl_{cfg}"
    mfe_col  = f"MFE_{cfg}"
    freq_col = f"freq_MFE_{cfg}"
    div_col  = f"ens_div_{cfg}"
    gquad_col     = f"gquad_present_{cfg}"
    mfe_delta_col = f"MFE_delta_gquad_{cfg}"
    psi_col  = f"predicted_PSI_{cfg}"

    missing = [c for c in [kl_col, mfe_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for config {cfg!r}: {missing}. Run enrich.py first.")

    # Collect continuous secondary features that have data
    secondary: list[tuple[str, str]] = []
    for col, label in [
        (freq_col,     "Freq MFE"),
        (div_col,      "Ensemble diversity"),
        (mfe_delta_col, "MFE delta G-quad (kcal/mol)"),
    ]:
        if col in df.columns and df[col].notna().any():
            secondary.append((col, label))
        else:
            print(f"  Skipping {col} (missing or all-NaN)")

    df = df.dropna(subset=[kl_col, mfe_col]).copy()
    print(f"  {len(df):,} rows with kl + MFE for {cfg}")

    # All continuous features used in Section 1 and Section 2
    feature_meta: list[tuple[str, str]] = [(mfe_col, "MFE (kcal/mol)")] + secondary

    # ── Existing supplementary plots ──────────────────────────────────────────
    def save(fig, name: str) -> None:
        _save(fig, out_dir, name)

    save(P.hist_kl(df, kl_col), "hist_kl.png")
    save(P.kl_curve(), "kl_curve.png")
    save(P.boxplot_kl_panels(df, kl_col), "boxplot_kl.png")

    for col, label in feature_meta:
        save(P.violin_kl_by_feature(df, col, kl_col, bins=bins, log_scale=False, feature_label=label),
             f"violin_{col}.png")
        save(P.violin_kl_by_feature(df, col, kl_col, bins=bins, log_scale=True, feature_label=label),
             f"violin_{col}_log.png")
        save(P.proportion_above_threshold(df, col, kl_col, bins=bins, feature_label=label),
             f"proportion_{col}.png")

    # PSI density plot
    if psi_col in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        P.density_histogram(ax, df["PSI"].values, df[psi_col].values,
                            xlabel="PSI (true)", ylabel=f"PSI ({cfg} predicted)",
                            title=f"PSI density | {cfg}")
        save(fig, "density_PSI_vs_pred.png")

    # Feature-vs-feature density plots
    for col, label in secondary:
        fig, ax = plt.subplots(figsize=(6, 5))
        P.density_histogram(ax, df[mfe_col].values, df[col].values,
                            xlabel="MFE (kcal/mol)", ylabel=label,
                            title=f"Feature density | {cfg}")
        save(fig, f"density_{mfe_col}_{col}.png")

    # gquad_present grouped bar (if available)
    if gquad_col in df.columns and df[gquad_col].notna().any():
        fig, ax = plt.subplots(figsize=(5, 4))
        groups = {False: df.loc[df[gquad_col] == False, kl_col].dropna().values,
                  True:  df.loc[df[gquad_col] == True,  kl_col].dropna().values}
        labels = ["G-quad absent", "G-quad present"]
        means  = [groups[False].mean() if len(groups[False]) else np.nan,
                  groups[True].mean()  if len(groups[True])  else np.nan]
        p90s   = [np.percentile(groups[False], 90) if len(groups[False]) else np.nan,
                  np.percentile(groups[True],  90) if len(groups[True])  else np.nan]
        x = np.arange(2)
        ax.bar(x - 0.2, means, 0.35, label="Mean KL", color=P._VIOLIN_FACE, edgecolor=P._VIOLIN_EDGE)
        ax.bar(x + 0.2, p90s,  0.35, label="P90 KL",  color="#e8622c",        edgecolor="#a04010", alpha=0.8)
        for xi, gv in enumerate([False, True]):
            n = len(groups[gv])
            ax.text(xi, max(means[xi] or 0, p90s[xi] or 0) * 1.02, f"n={n}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("KL")
        ax.set_title(f"KL by G-quadruplex presence | {cfg}")
        ax.legend()
        fig.tight_layout()
        save(fig, "kl_by_gquad.png")

    # ── Section 1: 1D binned line plots (full data) ───────────────────────────
    print("  [Section 1] 1D binned line plots …")
    _section1(df, feature_meta, kl_col, out_dir, bins)

    # ── Section 2: 2D heatmaps (full data) ───────────────────────────────────
    print("  [Section 2] 2D mean-KL heatmaps …")
    _section2(df, feature_meta, kl_col, out_dir, cfg)

    # ── Section 3: Restricted to bottom 20% MFE (most negative) ──────────────
    mfe_q20 = df[mfe_col].quantile(0.20)
    df_r    = df[df[mfe_col] <= mfe_q20].copy()
    print(f"  [Section 3] Restricted to MFE ≤ {mfe_q20:.2f} — {len(df_r):,} rows "
          f"({100*len(df_r)/len(df):.0f}%)")

    r_dir = out_dir / "restricted"
    r_dir.mkdir(parents=True, exist_ok=True)

    # Drop any secondary features that lost all data in the restricted subset
    feature_meta_r = [
        (col, label) for col, label in feature_meta
        if df_r[col].notna().any()
    ]
    _section1(df_r, feature_meta_r, kl_col, r_dir, bins, suffix="_restricted")
    _section2(df_r, feature_meta_r, kl_col, r_dir, cfg, suffix="_restricted")

    # ── stats.csv ─────────────────────────────────────────────────────────────
    kl = df[kl_col].values
    records = [{"config": cfg, "subset": "full", "metric": "kl",
                "mean": kl.mean(), "median": np.median(kl),
                "p90": np.percentile(kl, 90), "p95": np.percentile(kl, 95), "n": len(kl)}]
    for col, label in feature_meta:
        vals = df[col].values
        records.append({"config": cfg, "subset": "full", "metric": col,
                        "mean": vals.mean(), "median": np.median(vals),
                        "p90": np.percentile(vals, 90), "p95": np.percentile(vals, 95),
                        "n": len(vals)})
    kl_r = df_r[kl_col].values
    records.append({"config": cfg, "subset": "restricted_mfe_q20", "metric": "kl",
                    "mean": kl_r.mean(), "median": np.median(kl_r),
                    "p90": np.percentile(kl_r, 90), "p95": np.percentile(kl_r, 95),
                    "n": len(kl_r)})

    stats_path = out_dir / "stats.csv"
    pd.DataFrame(records).to_csv(stats_path, index=False)
    print(f"  Saved {stats_path.name}")

    # ── summary.md ────────────────────────────────────────────────────────────
    lines = [
        f"# Analysis: {cfg}\n\n",
        f"**Exons (full):** {len(df):,}  \n",
        f"**KL mean:** {kl.mean():.6f}  **KL median:** {np.median(kl):.6f}  "
        f"**KL p90:** {np.percentile(kl, 90):.6f}\n\n",
        f"**Restricted subset (MFE ≤ {mfe_q20:.2f}):** {len(df_r):,} rows  \n",
        f"**KL mean (restricted):** {kl_r.mean():.6f}  "
        f"**KL p90 (restricted):** {np.percentile(kl_r, 90):.6f}\n\n",
        "## Plots (full data)\n",
    ]
    for png in sorted(p for p in out_dir.glob("*.png")):
        lines.append(f"![{png.stem}]({png.name})\n\n")
    lines.append("## Plots (restricted — bottom 20% MFE)\n")
    for png in sorted(r_dir.glob("*.png")):
        lines.append(f"![{png.stem}](restricted/{png.name})\n\n")
    (out_dir / "summary.md").write_text("".join(lines))
    print("  Saved summary.md")


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
