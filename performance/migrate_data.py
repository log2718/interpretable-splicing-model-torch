"""One-time migration: merge 4 existing test CSVs into a single unified CSV.

Reads:
  data/test_data_rna_structure.csv  → base + baseline config columns
  data/test_vienna60_annotated.csv  → vienna60 config columns
  data/test_flank_40_30.csv         → flank_40_30 config columns

Writes:
  data/test_annotated.csv

Column naming convention: {metric}_{config_name}
  predicted_PSI_{config}, kl_{config}, MFE_{config},
  struct_{config}, freq_MFE_{config}, ens_div_{config}

Notes:
  - freq_MFE_vienna60 / ens_div_vienna60 are left as NaN.
    Run enrich.py --config-name vienna60 --temperature 60 --rnafold-only to fill them.
  - flanks_added is dropped (always True, carries no information).
  - All source CSVs have identical row order (same 47,962 exons); merge is positional.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SRC_BASELINE = DATA_DIR / "test_data_rna_structure.csv"
SRC_VIENNA60 = DATA_DIR / "test_vienna60_annotated.csv"
SRC_FLANK    = DATA_DIR / "test_flank_40_30.csv"
OUT_PATH     = DATA_DIR / "test_annotated.csv"

# ── Column specs ──────────────────────────────────────────────────────────────

BASE_COLS = [
    "exon", "num_DNA_reads", "num_exon_inclusion", "num_exon_skipping",
    "num_intron_retention", "num_splicing_in_exon", "num_bad_exon1",
    "num_bad_reads", "num_unknown_splicing", "total_noncanonical", "total", "PSI",
]

BASELINE_RENAME = {
    "predicted_PSI":    "predicted_PSI_baseline",
    "predicted_ss":     "struct_baseline",
    "predicted_mfe":    "MFE_baseline",
    "st_predicted_prob":"struct_prob_baseline",
    "freq_MFE":         "freq_MFE_baseline",
    "ensemble_diversity":"ens_div_baseline",
    "kl":               "kl_baseline",
}

VIENNA60_RENAME = {
    "predicted_secondary_struct": "struct_vienna60",
    "predicted_MFE":              "MFE_vienna60",
    "model_sequence":             "model_seq_vienna60",
    "wobble_count":               "wobble_count_vienna60",
    "predicted_PSI_60":           "predicted_PSI_vienna60",
    "kl_60":                      "kl_vienna60",
}

FLANK_RENAME = {
    "predicted_secondary_struct": "struct_flank_40_30",
    "predicted_MFE":              "MFE_flank_40_30",
    "model_sequence":             "model_seq_flank_40_30",
    "wobble_count":               "wobble_count_flank_40_30",
    "predicted_PSI_flanks":       "predicted_PSI_flank_40_30",
    "kl_flanks":                  "kl_flank_40_30",
    "freq_MFE":                   "freq_MFE_flank_40_30",
    "ensemble_diversity":         "ens_div_flank_40_30",
}


def main() -> None:
    for src in [SRC_BASELINE, SRC_VIENNA60, SRC_FLANK]:
        if not src.exists():
            raise FileNotFoundError(f"Source CSV not found: {src}")

    print("Reading source CSVs...")
    df_base    = pd.read_csv(SRC_BASELINE)
    df_vienna  = pd.read_csv(SRC_VIENNA60)
    df_flank   = pd.read_csv(SRC_FLANK)

    n = len(df_base)
    for df, name in [(df_vienna, "vienna60"), (df_flank, "flank_40_30")]:
        if len(df) != n:
            raise ValueError(f"{name} CSV has {len(df)} rows; expected {n}")
    print(f"  All CSVs: {n:,} rows ✓")

    # ── Base + baseline ───────────────────────────────────────────────────────
    baseline_cols = BASE_COLS + list(BASELINE_RENAME.keys())
    out = df_base[baseline_cols].rename(columns=BASELINE_RENAME).reset_index(drop=True)

    # ── Vienna60 config ───────────────────────────────────────────────────────
    v60_cols = df_vienna[list(VIENNA60_RENAME.keys())].rename(columns=VIENNA60_RENAME).reset_index(drop=True)
    # freq_MFE and ens_div for vienna60 were computed at 37°C (same as baseline) — set to NaN
    v60_cols["freq_MFE_vienna60"] = np.nan
    v60_cols["ens_div_vienna60"]  = np.nan
    out = pd.concat([out, v60_cols], axis=1)

    # ── Flank_40_30 config ────────────────────────────────────────────────────
    fl_cols = df_flank[list(FLANK_RENAME.keys())].rename(columns=FLANK_RENAME).reset_index(drop=True)
    out = pd.concat([out, fl_cols], axis=1)

    # ── Write ─────────────────────────────────────────────────────────────────
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"  Shape: {out.shape}")
    print(f"  Columns ({len(out.columns)}):")
    for col in out.columns:
        print(f"    {col}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\nSanity checks:")
    print(f"  kl_baseline  mean: {out['kl_baseline'].mean():.6f}")
    print(f"  kl_vienna60  mean: {out['kl_vienna60'].mean():.6f}")
    print(f"  kl_flank_40_30 mean: {out['kl_flank_40_30'].mean():.6f}")
    print(f"  freq_MFE_vienna60 NaN count: {out['freq_MFE_vienna60'].isna().sum()} (expected {n})")
    print("\nDone. Run enrich.py --config-name vienna60 --temperature 60 --rnafold-only to fill NaN ensemble features.")


if __name__ == "__main__":
    main()
