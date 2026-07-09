"""Flexible RNAduplex exon-intron interaction pipeline.

Allows configuring:
  - Which chunks of the flanking sequences to test (name, flank side, start, end)
  - Which portion of the exon to use: whole / first_half / second_half
  - Output file name

Chunks are defined via --chunks as: name:flank:start:end
  flank = left  → slice of LEFT_FLANK_150
  flank = right → slice of RIGHT_FLANK_100

Examples:
  # Original 4-chunk setup, whole exon:
  python performance/rnaduplex_interact_flexible.py \\
      --exon-portion whole \\
      --chunks up_far:left:0:50 up_mid:left:50:100 up_near:left:100:150 down:right:0:30 \\
      --out data/test_rnaduplex_whole_4chunk.csv

  # Upstream combined (150bp) + downstream 30bp, first half of exon:
  python performance/rnaduplex_interact_flexible.py \\
      --exon-portion first_half \\
      --chunks up_combined:left:0:150 down:right:0:30 \\
      --out data/test_rnaduplex_firsthalf_combined.csv

  # Downstream extended to 100bp, whole exon:
  python performance/rnaduplex_interact_flexible.py \\
      --exon-portion whole \\
      --chunks down_100:right:0:100 \\
      --out data/test_rnaduplex_down100.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# ── Flank sequences ───────────────────────────────────────────────────────────

LEFT_FLANK_150 = (
    "GGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACT"
    "GGGCATATGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATGTCTTTC"
    "TCTGCCATCCAGGTT"
)

RIGHT_FLANK_100 = "CAGGTCTGACTATGGGACCCTTGATGTTTTCTTTCCCCTTCTTTTCTATGGTTAAGTTCATGTCATAGGAAGGGGAGAAGTAACAGGGTACAGTTTAGAA"

# Downstream 250nt — RIGHT_FLANK_100 is the first 100nt of this
DOWNSTREAM_LONG = "CAGGTCTGACTATGGGACCCTTGATGTTTTCTTTCCCCTTCTTTTCTATGGTTAAGTTCATGTCATAGGAAGGGGAGAAGTAACAGGGTACAGTTTAGAATGGGAAACAGACGAATGATTGCATCAGTGTGGAAGTCTCAGGATCGTTTTAGTTTCTTTTATTTGCTGTTCATAACAATTGTTTTCTTTTGTTTAATTCTTGCTTTCTTTTTTTTTCTTCTCCGCAATTTTTACTATTATACTTAATGCC"

# Downstream exon — 141nt exon located ~853nt downstream of the variable exon
DOWNSTREAM_EXON = "CTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCACTAAGCTCGCTCTAGA"

# Fixed upstream exon — kept constant across all MPRA experiments
UPSTREAM_EXON = "GTACCGCAACCTCAAACAGACACCATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAG"

# Mononucleotide shuffle of upstream exon (seed=42) — preserves GC content, negative control
UPSTREAM_EXON_SHUFFLED = "CTCGCTCCGAGACCAAGTGTGTTTGGAGCTTACATTAAAGTGTGGCGGGGACGCGCGGAGATATGGCAGCCGGCGCAAACTAGCGGTTCTCGAGACGTGTCAGCGCTCAACAGCGA"

_FLANKS = {
    "left":                   LEFT_FLANK_150,
    "right":                  RIGHT_FLANK_100,
    "downstream_long":        DOWNSTREAM_LONG,
    "downstream_exon":        DOWNSTREAM_EXON,
    "upstream_exon":          UPSTREAM_EXON,
    "upstream_exon_shuffled": UPSTREAM_EXON_SHUFFLED,
}

_MFE_RE    = re.compile(r"\(\s*(-?[0-9]+\.?[0-9]*)\s*\)\s*$")
_STRUCT_RE = re.compile(r"^[.()\[\]{}<>]")

_TEMPERATURE:   float       = 37.0
_RNADUPLEX_BIN: str         = "RNAduplex"
_CHUNKS:        dict        = {}
_EXON_PORTION:  str         = "whole"


def _to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _get_exon_portion(exon: str, portion: str) -> str:
    n = len(exon)
    if portion == "first_half":
        return exon[: n // 2]
    if portion == "second_half":
        return exon[n // 2 :]
    return exon  # whole


def _run_rnaduplex(exon_rna: str, chunk_rna: str) -> tuple[int, float]:
    query = f">exon\n{exon_rna}\n>chunk\n{chunk_rna}\n"
    try:
        result = subprocess.run(
            [_RNADUPLEX_BIN, "-T", str(_TEMPERATURE)],
            input=query, text=True, capture_output=True, check=True,
        )
    except subprocess.CalledProcessError:
        return 0, float("nan")

    for line in result.stdout.splitlines():
        line = line.strip()
        if "&" in line and _STRUCT_RE.match(line):
            mfe_match = _MFE_RE.search(line)
            mfe = float(mfe_match.group(1)) if mfe_match else float("nan")
            struct_exon = line.split("&")[0]
            n_pairs = struct_exon.count("(")
            return n_pairs, mfe

    return 0, float("nan")


def _worker(exon: str) -> dict:
    portion = _get_exon_portion(exon, _EXON_PORTION)
    exon_rna = _to_rna(portion)
    row: dict = {"exon": exon}
    for name, chunk_seq in _CHUNKS.items():
        chunk_rna = _to_rna(chunk_seq)
        n, mfe = _run_rnaduplex(exon_rna, chunk_rna)
        row[f"n_pairs_{name}"] = n
        row[f"mfe_{name}"]     = mfe
    return row


def _init_worker(temperature: float, rnaduplex_bin: str,
                 chunks: dict, exon_portion: str) -> None:
    global _TEMPERATURE, _RNADUPLEX_BIN, _CHUNKS, _EXON_PORTION
    _TEMPERATURE   = temperature
    _RNADUPLEX_BIN = rnaduplex_bin
    _CHUNKS        = chunks
    _EXON_PORTION  = exon_portion


def _parse_chunks(chunk_specs: list[str]) -> dict[str, str]:
    """Parse 'name:flank:start:end' strings into {name: sequence}."""
    chunks = {}
    for spec in chunk_specs:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(f"Bad chunk spec '{spec}' — expected name:flank:start:end")
        name, flank, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
        if flank not in _FLANKS:
            raise ValueError(f"flank must be 'left' or 'right', got '{flank}'")
        seq = _FLANKS[flank][start:end]
        if "N" in seq.upper():
            raise ValueError(
                f"Chunk '{name}' contains N — RIGHT_FLANK_100 beyond position 30 "
                f"has not been filled in. Edit RIGHT_FLANK_100 in this script first."
            )
        chunks[name] = seq
    return chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",           type=Path,   default=BASE / "data/test_annotated.csv")
    p.add_argument("--out",           type=Path,   required=True,
                   help="Output CSV path, e.g. data/test_rnaduplex_custom.csv")
    p.add_argument("--chunks",        nargs="+",   required=True,
                   metavar="name:flank:start:end",
                   help="Chunk definitions. flank=left|right, positions in bp.")
    p.add_argument("--exon-portion",  choices=["whole", "first_half", "second_half"],
                   default="whole",
                   help="Which portion of the exon to use as query.")
    p.add_argument("--workers",       type=int,    default=1)
    p.add_argument("--temperature",   type=float,  default=37.0)
    p.add_argument("--rnaduplex-bin", type=str,    default="RNAduplex")
    p.add_argument("--force",         action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists — skipping (use --force to recompute).")
        return

    chunks = _parse_chunks(args.chunks)

    print(f"Exon portion : {args.exon_portion}")
    print(f"Chunks       :")
    for name, seq in chunks.items():
        print(f"  {name:<15} ({len(seq)} nt)  {seq}")

    df    = pd.read_csv(args.csv, usecols=["exon"])
    exons = df["exon"].tolist()
    print(f"\nLoaded {len(exons):,} exons")
    print(f"Temperature: {args.temperature}°C   Workers: {args.workers}")
    print(f"Total RNAduplex calls: {len(exons) * len(chunks):,}")

    rows: list[dict] = []
    if args.workers > 1:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.temperature, args.rnaduplex_bin, chunks, args.exon_portion),
        ) as pool:
            for row in tqdm(pool.imap(_worker, exons, chunksize=32),
                            total=len(exons), desc="RNAduplex", unit="exon"):
                rows.append(row)
    else:
        _init_worker(args.temperature, args.rnaduplex_bin, chunks, args.exon_portion)
        for exon in tqdm(exons, desc="RNAduplex", unit="exon"):
            rows.append(_worker(exon))

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  ({len(out_df):,} rows, {len(out_df.columns)} cols)")

    for name in chunks:
        col = f"mfe_{name}"
        v   = out_df[col]
        print(f"  {col}: mean={v.mean():.2f}  min={v.min():.2f}  "
              f"bottom2%={v.quantile(0.02):.2f}")


if __name__ == "__main__":
    main()
