"""CLI for preparing model-ready datasets from sequence CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import DEFAULT_SEQUENCE_COLUMN, dataframe_to_dataset, save_dataset_npz


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        Configured argument parser for dataset preparation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read a CSV of unflanked exon sequences, compute structure and "
            "wobble features, and save a compressed NPZ dataset."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV file containing an unflanked sequence column.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output .npz path for the prepared dataset.",
    )
    parser.add_argument(
        "--sequence-column",
        default=DEFAULT_SEQUENCE_COLUMN,
        help=(
            "Column name for the unflanked input sequence. Defaults to "
            f"{DEFAULT_SEQUENCE_COLUMN!r}."
        ),
    )
    parser.add_argument(
        "--no-flanks",
        action="store_true",
        help="Skip adding the fixed model flanks before computing features.",
    )
    parser.add_argument(
        "--left-flank",
        default="",
        help="Override the left flank sequence (default: utils.LEFT_FLANK).",
    )
    parser.add_argument(
        "--right-flank",
        default="",
        help="Override the right flank sequence (default: utils.RIGHT_FLANK).",
    )
    parser.add_argument(
        "--rnafold-bin",
        default="RNAfold",
        help="Executable name or path for ViennaRNA RNAfold.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=37.0,
        help="RNAfold temperature in Celsius.",
    )
    parser.add_argument(
        "--max-bp-span",
        type=int,
        default=0,
        help="Optional RNAfold maximum base-pair span. Zero disables the flag.",
    )
    parser.add_argument(
        "--commands-file",
        default="",
        help="Optional ViennaRNA commands file passed through to RNAfold.",
    )
    parser.add_argument(
        "--num-threads",
        default=8,
        type=int,
        help="Number of threads to use for ViennaRNA.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help=(
            "Optional output CSV path. If provided, save an annotated CSV "
            "containing original columns plus predicted secondary structure, "
            "MFE, and other structure-related annotations."
        ),
    )
    parser.add_argument(
        "--gquad",
        action="store_true",
        help=(
            "Also run RNAfold -g to compute G-quadruplex features: "
            "gquad_present (bool) and MFE_delta_gquad (kcal/mol). "
            "Doubles RNAfold calls but adds no other overhead."
        ),
    )
    return parser


def _build_annotated_csv(
    df: pd.DataFrame, dataset: dict, sequence_column: str
) -> pd.DataFrame:
    """Merge original CSV columns with structure annotations from the dataset.

    Args:
        df: The original input dataframe.
        dataset: Dataset dictionary produced by ``dataframe_to_dataset``.
        sequence_column: Name of the sequence column used during preprocessing.

    Returns:
        An annotated dataframe ready for CSV export.
    """
    import numpy as np

    out = df.copy()

    # Core structure predictions
    out["predicted_secondary_struct"] = dataset["structure"].tolist()
    out["predicted_MFE"] = dataset["mfe"].tolist()

    # Flanked model sequence (useful for reproducing predictions)
    out["model_sequence"] = dataset["model_sequence"].tolist()

    # Per-position wobble sums (wobbles shape: N×1×L)
    wobble_arr = dataset["wobbles"]
    out["wobble_count"] = wobble_arr.reshape(wobble_arr.shape[0], -1).sum(axis=1).astype(int)

    # Whether flanks were added
    out["flanks_added"] = bool(dataset["added_flanks"])

    # G-quadruplex features (only present when --gquad was used)
    if "metadata_gquad_present" in dataset:
        out["gquad_present"]    = dataset["metadata_gquad_present"].tolist()
        out["MFE_delta_gquad"]  = dataset["metadata_MFE_delta_gquad"].tolist()

    return out


def main() -> None:
    """Run the dataset preparation CLI."""
    args = build_parser().parse_args()

    df = pd.read_csv(args.input_csv)
    dataset = dataframe_to_dataset(
        df,
        sequence_column=args.sequence_column,
        add_flanks=not args.no_flanks,
        left_flank=args.left_flank or None,
        right_flank=args.right_flank or None,
        rnafold_bin=args.rnafold_bin,
        temperature=args.temperature,
        maxBPspan=args.max_bp_span,
        commands_file=args.commands_file,
        num_threads=args.num_threads,
        gquad=args.gquad,
    )

    # ── Save NPZ (always) ─────────────────────────────────────────────────
    output_path = save_dataset_npz(dataset, args.output_path)

    print(f"Saved dataset to {Path(output_path).resolve()}")
    print(f"Examples: {dataset['seq_oh'].shape[0]}")
    print(f"Sequence tensor shape (N, 4, L): {dataset['seq_oh'].shape}")
    print(f"Structure tensor shape (N, 3, L): {dataset['struct_oh'].shape}")
    print(f"Wobble tensor shape (N, 1, L): {dataset['wobbles'].shape}")
    print("Feature arrays are NumPy arrays; convert them to torch tensors for model inference.")

    # ── Save annotated CSV (optional) ─────────────────────────────────────
    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_df = _build_annotated_csv(df, dataset, args.sequence_column)
        annotated_df.to_csv(csv_path, index=False)
        print(f"Saved annotated CSV to {csv_path.resolve()}  ({len(annotated_df)} rows)")


if __name__ == "__main__":
    main()
