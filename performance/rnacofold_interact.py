"""Compute RNAcofold exon-intron interaction features for the flank_150_30 config.

For each exon, runs RNAcofold(exon & chunk) for 4 fixed chunks of the flanking
sequences and extracts:
  - n_pairs    : number of intermolecular base pairs
                 (unmatched '(' in the exon half of the dot-bracket — these
                  must pair with ')' in the chunk half, i.e. cross-molecule pairs)
  - cofold_mfe : raw cofold MFE in kcal/mol

Chunks (all fixed — only the exon varies per row):
  up_dist  = LEFT_FLANK_150[  0: 50]   distal upstream,   100-150 nt from exon
  up_mid   = LEFT_FLANK_150[ 50:100]   middle upstream,    50-100 nt from exon
  up_prox  = LEFT_FLANK_150[100:150]   proximal upstream,   0-50  nt from exon (3'SS)
  down     = RIGHT_FLANK_30 [  0: 30]  downstream,          0-30  nt from exon (5'SS)

Output: data/test_rnacofold_interact.csv
Columns: exon,
         n_pairs_up_dist, mfe_up_dist,
         n_pairs_up_mid,  mfe_up_mid,
         n_pairs_up_prox, mfe_up_prox,
         n_pairs_down,    mfe_down

Usage:
    python performance/rnacofold_interact.py \\
        --csv data/test_annotated.csv \\
        --out data/test_rnacofold_interact.csv \\
        --workers 8 \\
        --temperature 37
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

# ── Fixed flanking sequences for flank_150_30 ─────────────────────────────────
# Verified from data/test_flank_150_30.npz: same for all 47,962 examples.

LEFT_FLANK_150 = (
    "GGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACT"
    "GGGCATATGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATGTCTTTC"
    "TCTGCCATCCAGGTT"
)
RIGHT_FLANK_30 = "CAGGTCTGACTATGGGACCCTTGATGTTTT"

CHUNKS = {
    "up_dist":  LEFT_FLANK_150[  0: 50],
    "up_mid":   LEFT_FLANK_150[ 50:100],
    "up_prox":  LEFT_FLANK_150[100:150],
    "down":     RIGHT_FLANK_30[  0: 30],
}

# ── Regex to parse MFE from cofold output ─────────────────────────────────────
# MFE is always the last token on the structure line: "STRUCT&STRUCT   (-7.20)"
# Require at least one digit and anchor to end of line to avoid matching dots in structure.
_MFE_RE   = re.compile(r"\(\s*(-?[0-9]+\.?[0-9]*)\s*\)\s*$")
_STRUCT_RE = re.compile(r"^[.()\[\]{}<>]")


def _to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _count_intermolecular(struct_exon: str) -> int:
    """Count unmatched '(' in the exon half — these pair with the chunk (intermolecular)."""
    depth = 0
    for c in struct_exon:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
    return max(depth, 0)


def _run_cofold(exon_rna: str, chunk_rna: str, temperature: float,
                rnacofold_bin: str) -> tuple[int, float]:
    """Run RNAcofold and return (n_pairs, cofold_mfe)."""
    query = exon_rna + "&" + chunk_rna
    try:
        result = subprocess.run(
            [rnacofold_bin, "--noPS", "-T", str(temperature)],
            input=query + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return 0, float("nan")

    for line in result.stdout.splitlines():
        line = line.strip()
        if "&" in line and _STRUCT_RE.match(line):
            # line looks like: "struct1&struct2   (-5.30)"
            mfe_match = _MFE_RE.search(line)
            mfe = float(mfe_match.group(1)) if mfe_match else float("nan")

            # struct part is everything before the whitespace+MFE at the end
            tokens = line.rsplit(None, 1)
            struct_exon = tokens[0].split("&")[0] if len(tokens) >= 1 else ""
            n_pairs = _count_intermolecular(struct_exon)
            return n_pairs, mfe

    return 0, float("nan")


# ── Worker function (top-level so multiprocessing can pickle it) ──────────────

_TEMPERATURE: float = 37.0
_RNACOFOLD_BIN: str = "RNAcofold"


def _worker(exon: str) -> dict:
    exon_rna = _to_rna(exon)
    row: dict = {"exon": exon}
    for name, chunk in CHUNKS.items():
        chunk_rna = _to_rna(chunk)
        n, mfe = _run_cofold(exon_rna, chunk_rna, _TEMPERATURE, _RNACOFOLD_BIN)
        row[f"n_pairs_{name}"] = n
        row[f"mfe_{name}"] = mfe
    return row


def _init_worker(temperature: float, rnacofold_bin: str) -> None:
    global _TEMPERATURE, _RNACOFOLD_BIN
    _TEMPERATURE = temperature
    _RNACOFOLD_BIN = rnacofold_bin


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute RNAcofold exon-intron interaction features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",          type=Path, default=BASE / "data/test_annotated.csv")
    p.add_argument("--out",          type=Path, default=BASE / "data/test_rnacofold_interact.csv")
    p.add_argument("--workers",      type=int,  default=1)
    p.add_argument("--temperature",  type=float,default=37.0)
    p.add_argument("--rnacofold-bin",type=str,  default="RNAcofold")
    p.add_argument("--force",        action="store_true",
                   help="Recompute even if output CSV already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists — skipping (use --force to recompute).")
        return

    df = pd.read_csv(args.csv, usecols=["exon"])
    exons = df["exon"].tolist()
    print(f"Loaded {len(exons):,} exons from {args.csv}")
    print(f"Chunks: {list(CHUNKS.keys())} ({[len(v) for v in CHUNKS.values()]} nt each)")
    print(f"Temperature: {args.temperature}°C   Workers: {args.workers}")
    print(f"Total RNAcofold calls: {len(exons) * len(CHUNKS):,}")

    rows: list[dict] = []

    if args.workers > 1:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.temperature, args.rnacofold_bin),
        ) as pool:
            for row in tqdm(
                pool.imap(_worker, exons, chunksize=32),
                total=len(exons),
                desc="RNAcofold",
                unit="exon",
            ):
                rows.append(row)
    else:
        _init_worker(args.temperature, args.rnacofold_bin)
        for exon in tqdm(exons, desc="RNAcofold", unit="exon"):
            rows.append(_worker(exon))

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  ({len(out_df):,} rows, {len(out_df.columns)} cols)")

    # Quick summary
    for name in CHUNKS:
        col = f"n_pairs_{name}"
        vals = out_df[col]
        print(f"  {col}: mean={vals.mean():.2f}  max={vals.max()}  "
              f"top2%_threshold={vals.quantile(0.98):.0f}")


if __name__ == "__main__":
    main()
