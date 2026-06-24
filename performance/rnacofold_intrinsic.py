"""Compute RNAcofold intra-exon interaction (first half vs second half).

For each 70-nt exon, runs RNAcofold(exon[:35] & exon[35:]) and counts
intermolecular base pairs as a positive-control for exon structure effects.

Output: data/test_rnacofold_intrinsic.csv
Columns: exon, n_pairs_intrinsic, mfe_intrinsic

Usage:
    python performance/rnacofold_intrinsic.py \\
        --csv data/test_annotated.csv \\
        --out data/test_rnacofold_intrinsic.csv \\
        --workers 16 \\
        --temperature 37
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

_MFE_RE    = re.compile(r"\(\s*(-?[0-9]+\.?[0-9]*)\s*\)\s*$")
_STRUCT_RE = re.compile(r"^[.()\[\]{}<>]")

EXON_LEN  = 70
HALF      = EXON_LEN // 2   # 35


def _to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _count_intermolecular(struct_first: str) -> int:
    """Unmatched '(' in first-half structure = intermolecular pairs."""
    depth = 0
    for c in struct_first:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
    return max(depth, 0)


_TEMPERATURE:    float = 37.0
_RNACOFOLD_BIN:  str   = "RNAcofold"
_RNADUPLEX_BIN:  str   = "RNAduplex"


def _run_rnacofold(first_rna: str, second_rna: str) -> tuple[int, float]:
    query = first_rna + "&" + second_rna
    try:
        result = subprocess.run(
            [_RNACOFOLD_BIN, "--noPS", "-T", str(_TEMPERATURE)],
            input=query + "\n", text=True, capture_output=True, check=True,
        )
    except subprocess.CalledProcessError:
        return 0, float("nan")
    for line in result.stdout.splitlines():
        line = line.strip()
        if "&" in line and _STRUCT_RE.match(line):
            mfe_match = _MFE_RE.search(line)
            mfe = float(mfe_match.group(1)) if mfe_match else float("nan")
            tokens = line.rsplit(None, 1)
            struct_first = tokens[0].split("&")[0] if len(tokens) >= 1 else ""
            return _count_intermolecular(struct_first), mfe
    return 0, float("nan")


def _run_rnaduplex(first_rna: str, second_rna: str) -> tuple[int, float]:
    query = f">s1\n{first_rna}\n>s2\n{second_rna}\n"
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
            struct_first = line.split("&")[0]
            n_pairs = struct_first.count("(")
            return n_pairs, mfe
    return 0, float("nan")


def _worker(exon: str) -> dict:
    first  = _to_rna(exon[:HALF])
    second = _to_rna(exon[HALF:])

    n_cofold, mfe_cofold   = _run_rnacofold(first, second)
    n_duplex, mfe_duplex   = _run_rnaduplex(first, second)

    return {
        "exon":               exon,
        "n_pairs_intrinsic":  n_cofold,
        "mfe_intrinsic":      mfe_cofold,
        "n_pairs_rnaduplex":  n_duplex,
        "mfe_rnaduplex":      mfe_duplex,
    }


def _init_worker(temperature: float, rnacofold_bin: str, rnaduplex_bin: str) -> None:
    global _TEMPERATURE, _RNACOFOLD_BIN, _RNADUPLEX_BIN
    _TEMPERATURE   = temperature
    _RNACOFOLD_BIN = rnacofold_bin
    _RNADUPLEX_BIN = rnaduplex_bin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",            type=Path,  default=BASE / "data/test_annotated.csv")
    p.add_argument("--out",            type=Path,  default=BASE / "data/test_rnacofold_intrinsic.csv")
    p.add_argument("--workers",        type=int,   default=1)
    p.add_argument("--temperature",    type=float, default=37.0)
    p.add_argument("--rnacofold-bin",  type=str,   default="RNAcofold")
    p.add_argument("--rnaduplex-bin",  type=str,   default="RNAduplex")
    p.add_argument("--force",          action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists — skipping (use --force to recompute).")
        return

    df    = pd.read_csv(args.csv, usecols=["exon"])
    exons = df["exon"].tolist()
    print(f"Loaded {len(exons):,} exons  |  split at position {HALF}")
    print(f"Temperature: {args.temperature}°C   Workers: {args.workers}")

    rows: list[dict] = []
    if args.workers > 1:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.temperature, args.rnacofold_bin, args.rnaduplex_bin),
        ) as pool:
            for row in tqdm(pool.imap(_worker, exons, chunksize=32),
                            total=len(exons), desc="RNAcofold+RNAduplex intrinsic", unit="exon"):
                rows.append(row)
    else:
        _init_worker(args.temperature, args.rnacofold_bin, args.rnaduplex_bin)
        for exon in tqdm(exons, desc="RNAcofold intrinsic", unit="exon"):
            rows.append(_worker(exon))

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  ({len(out_df):,} rows)")

    for col in ["n_pairs_intrinsic", "mfe_intrinsic", "n_pairs_rnaduplex", "mfe_rnaduplex"]:
        v = out_df[col]
        print(f"  {col}: mean={v.mean():.2f}  min={v.min():.2f}  "
              f"bottom2%={v.quantile(0.02):.2f}")


if __name__ == "__main__":
    main()
