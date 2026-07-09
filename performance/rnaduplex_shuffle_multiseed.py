"""Run RNAduplex with N mononucleotide-shuffled versions of each chunk.

For each exon, runs RNAduplex against all N shuffled sequences in one pass.
Output has columns: mfe_{chunk}_seed_{i}, n_pairs_{chunk}_seed_{i}

Usage:
  # Upstream exon, 10 seeds
  python performance/rnaduplex_shuffle_multiseed.py \\
      --chunks upstream_exon:upstream_exon:0:116 \\
      --n-seeds 10 \\
      --out data/test_rnaduplex_upstream_exon_multiseed.csv \\
      --workers 8

  # 4 intron chunks, 10 seeds (HPC recommended: 4*10*48k = 1.9M calls)
  python performance/rnaduplex_shuffle_multiseed.py \\
      --chunks up_far:left:0:50 up_mid:left:50:100 up_near:left:100:150 down:right:0:30 \\
      --n-seeds 10 \\
      --out data/test_rnaduplex_intron_chunks_multiseed.csv \\
      --workers 8
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
RIGHT_FLANK_100 = "CAGGTCTGACTATGGGACCCTTGATGTTTTCTTTCCCCTTCTTTTCTATGGTTAAGTTCATGTCATAGGAAGGGGAGAAGTAACAGGGTACAGTTTAGAA"
UPSTREAM_EXON   = "GTACCGCAACCTCAAACAGACACCATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAG"

_FLANKS = {
    "left":          LEFT_FLANK_150,
    "right":         RIGHT_FLANK_100,
    "upstream_exon": UPSTREAM_EXON,
}

_MFE_RE    = re.compile(r"\(\s*(-?[0-9]+\.?[0-9]*)\s*\)\s*$")
_STRUCT_RE = re.compile(r"^[.()\[\]{}<>]")

_TEMPERATURE:   float = 37.0
_RNADUPLEX_BIN: str   = "RNAduplex"
_SHUFFLED:      dict  = {}   # {chunk_name: [seq_seed0, seq_seed1, ...]}


def _to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _shuffle(seq: str, rng: np.random.Generator) -> str:
    arr = list(seq)
    rng.shuffle(arr)
    return "".join(arr)


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
            n_pairs = line.split("&")[0].count("(")
            return n_pairs, mfe
    return 0, float("nan")


def _worker(exon: str) -> dict:
    exon_rna = _to_rna(exon)
    row: dict = {"exon": exon}
    for name, seqs in _SHUFFLED.items():
        for seed_i, seq in enumerate(seqs):
            n, mfe = _run_rnaduplex(exon_rna, _to_rna(seq))
            row[f"n_pairs_{name}_seed_{seed_i}"] = n
            row[f"mfe_{name}_seed_{seed_i}"]     = mfe
    return row


def _init_worker(temperature: float, rnaduplex_bin: str, shuffled: dict) -> None:
    global _TEMPERATURE, _RNADUPLEX_BIN, _SHUFFLED
    _TEMPERATURE   = temperature
    _RNADUPLEX_BIN = rnaduplex_bin
    _SHUFFLED      = shuffled


def _parse_chunks(chunk_specs: list[str]) -> dict[str, str]:
    chunks = {}
    for spec in chunk_specs:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(f"Bad chunk spec '{spec}' — expected name:flank:start:end")
        name, flank, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
        if flank not in _FLANKS:
            raise ValueError(f"flank must be one of {list(_FLANKS)}, got '{flank}'")
        chunks[name] = _FLANKS[flank][start:end]
    return chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",           type=Path,  default=BASE / "data/test_annotated.csv")
    p.add_argument("--out",           type=Path,  required=True)
    p.add_argument("--chunks",        nargs="+",  required=True,
                   metavar="name:flank:start:end")
    p.add_argument("--n-seeds",       type=int,   default=10)
    p.add_argument("--seed-start",    type=int,   default=0,
                   help="First seed value (seeds will be seed-start to seed-start+n-seeds-1)")
    p.add_argument("--workers",       type=int,   default=1)
    p.add_argument("--temperature",   type=float, default=37.0)
    p.add_argument("--rnaduplex-bin", type=str,   default="RNAduplex")
    p.add_argument("--force",         action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists — skipping (use --force to recompute).")
        return

    chunks = _parse_chunks(args.chunks)

    # Generate N shuffled versions of each chunk (seeds: seed_start to seed_start+n_seeds-1)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    shuffled: dict[str, list[str]] = {}
    print(f"Generating {args.n_seeds} shuffles per chunk (seeds {seeds[0]}-{seeds[-1]}):")
    for name, seq in chunks.items():
        shuffled[name] = [_shuffle(seq, np.random.default_rng(s)) for s in seeds]
        gc = (seq.upper().count("G") + seq.upper().count("C")) / len(seq)
        print(f"  {name:<20} ({len(seq)} nt, GC={gc:.2f})")
        for i, s in enumerate(shuffled[name]):
            print(f"    seed {i}: {s}")

    df    = pd.read_csv(args.csv, usecols=["exon"])
    exons = df["exon"].tolist()
    n_calls = len(exons) * sum(len(v) for v in shuffled.values())
    print(f"\nLoaded {len(exons):,} exons")
    print(f"Total RNAduplex calls: {n_calls:,}  "
          f"({len(chunks)} chunks × {args.n_seeds} seeds × {len(exons):,} exons)")

    rows: list[dict] = []
    if args.workers > 1:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.temperature, args.rnaduplex_bin, shuffled),
        ) as pool:
            for row in tqdm(pool.imap(_worker, exons, chunksize=32),
                            total=len(exons), desc="RNAduplex (multiseed)", unit="exon"):
                rows.append(row)
    else:
        _init_worker(args.temperature, args.rnaduplex_bin, shuffled)
        for exon in tqdm(exons, desc="RNAduplex (multiseed)", unit="exon"):
            rows.append(_worker(exon))

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  ({len(out_df):,} rows, {len(out_df.columns)} cols)")


if __name__ == "__main__":
    main()
