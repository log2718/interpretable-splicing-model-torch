"""Compute RNAduplex exon-intron features using SHUFFLED intron chunks.

Identical to rnaduplex_interact.py except each chunk sequence is
mononucleotide-shuffled (fixed seed) before any RNAduplex calls.
This is a negative control: if the MFE signal survives shuffling it is
driven by GC content alone; if it disappears the effect is
sequence-specific.

Output: data/test_rnaduplex_interact_shuffled.csv
        (same columns as test_rnaduplex_interact.csv)

Usage:
    python performance/rnaduplex_interact_shuffled.py \\
        --csv data/test_annotated.csv \\
        --out data/test_rnaduplex_interact_shuffled.csv \\
        --workers 8 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

LEFT_FLANK_150 = (
    "GGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACT"
    "GGGCATATGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATGTCTTTC"
    "TCTGCCATCCAGGTT"
)
RIGHT_FLANK_30 = "CAGGTCTGACTATGGGACCCTTGATGTTTT"

_REAL_CHUNKS = {
    "up_far":  LEFT_FLANK_150[  0: 50],
    "up_mid":  LEFT_FLANK_150[ 50:100],
    "up_near": LEFT_FLANK_150[100:150],
    "down":    RIGHT_FLANK_30[  0: 30],
}

_MFE_RE    = re.compile(r"\(\s*(-?[0-9]+\.?[0-9]*)\s*\)\s*$")
_STRUCT_RE = re.compile(r"^[.()\[\]{}<>]")


def _shuffle_chunk(seq: str, rng: np.random.Generator) -> str:
    arr = list(seq)
    rng.shuffle(arr)
    return "".join(arr)


def _to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


_TEMPERATURE:   float = 37.0
_RNADUPLEX_BIN: str   = "RNAduplex"
_SHUFFLED_CHUNKS: dict[str, str] = {}


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
    exon_rna = _to_rna(exon)
    row: dict = {"exon": exon}
    for name, chunk in _SHUFFLED_CHUNKS.items():
        chunk_rna = _to_rna(chunk)
        n, mfe = _run_rnaduplex(exon_rna, chunk_rna)
        row[f"n_pairs_{name}"] = n
        row[f"mfe_{name}"]     = mfe
    return row


def _init_worker(temperature: float, rnaduplex_bin: str,
                 shuffled_chunks: dict[str, str]) -> None:
    global _TEMPERATURE, _RNADUPLEX_BIN, _SHUFFLED_CHUNKS
    _TEMPERATURE      = temperature
    _RNADUPLEX_BIN    = rnaduplex_bin
    _SHUFFLED_CHUNKS  = shuffled_chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",           type=Path,  default=BASE / "data/test_annotated.csv")
    p.add_argument("--out",           type=Path,  default=BASE / "data/test_rnaduplex_interact_shuffled.csv")
    p.add_argument("--workers",       type=int,   default=1)
    p.add_argument("--temperature",   type=float, default=37.0)
    p.add_argument("--rnaduplex-bin", type=str,   default="RNAduplex")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--force",         action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists — skipping (use --force to recompute).")
        return

    rng = np.random.default_rng(args.seed)
    shuffled_chunks = {name: _shuffle_chunk(seq, rng)
                       for name, seq in _REAL_CHUNKS.items()}

    print("Shuffled chunks (seed={args.seed}):")
    for name in shuffled_chunks:
        real = _REAL_CHUNKS[name]
        shuf = shuffled_chunks[name]
        gc_real = (real.upper().count("G") + real.upper().count("C")) / len(real)
        gc_shuf = (shuf.upper().count("G") + shuf.upper().count("C")) / len(shuf)
        print(f"  {name:<8}  real: {real}  GC={gc_real:.2f}")
        print(f"  {name:<8}  shuf: {shuf}  GC={gc_shuf:.2f}")

    df    = pd.read_csv(args.csv, usecols=["exon"])
    exons = df["exon"].tolist()
    print(f"\nLoaded {len(exons):,} exons")
    print(f"Temperature: {args.temperature}°C   Workers: {args.workers}")
    print(f"Total RNAduplex calls: {len(exons) * len(shuffled_chunks):,}")

    rows: list[dict] = []
    if args.workers > 1:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.temperature, args.rnaduplex_bin, shuffled_chunks),
        ) as pool:
            for row in tqdm(pool.imap(_worker, exons, chunksize=32),
                            total=len(exons), desc="RNAduplex (shuffled)", unit="exon"):
                rows.append(row)
    else:
        _init_worker(args.temperature, args.rnaduplex_bin, shuffled_chunks)
        for exon in tqdm(exons, desc="RNAduplex (shuffled)", unit="exon"):
            rows.append(_worker(exon))

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  ({len(out_df):,} rows, {len(out_df.columns)} cols)")

    for name in shuffled_chunks:
        col = f"mfe_{name}"
        v   = out_df[col]
        print(f"  {col}: mean={v.mean():.2f}  min={v.min():.2f}  "
              f"bottom2%={v.quantile(0.02):.2f}")


if __name__ == "__main__":
    main()
