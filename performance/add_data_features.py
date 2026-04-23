from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))

from utils import LEFT_FLANK, RIGHT_FLANK

FEATURE_LINE_RE = re.compile(
	r"frequency of mfe structure in ensemble\s+([0-9eE+\-.]+);\s+"
	r"ensemble diversity\s+([0-9eE+\-.]+)"
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Add RNAfold-derived features to test_data.csv and write "
			"test_data_rna_structure.csv."
		)
	)
	parser.add_argument(
		"--input-csv",
		type=Path,
		default=BASE_DIR / "data" / "test_data.csv",
		help="Input CSV with an exon column.",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=BASE_DIR / "data" / "test_data_rna_structure.csv",
		help="Output CSV path.",
	)
	parser.add_argument(
		"--rnafold-bin",
		type=str,
		default="RNAfold",
		help="RNAfold executable name or full path.",
	)
	return parser.parse_args()


def check_rnafold_available(rnafold_bin: str) -> None:
	if Path(rnafold_bin).expanduser().exists():
		return
	if shutil.which(rnafold_bin) is not None:
		return
	raise FileNotFoundError(
		f"Could not find RNAfold executable {rnafold_bin!r}. "
		"Install ViennaRNA and make RNAfold available on PATH, "
		"or pass --rnafold-bin with an absolute path."
	)


def compute_rnafold_features(exon_sequence: str, rnafold_bin: str) -> tuple[float, float]:
	full_sequence = (LEFT_FLANK + exon_sequence + RIGHT_FLANK).upper().replace("T", "U")

	try:
		result = subprocess.run(
			[rnafold_bin, "-p", "--noPS"],
			input=full_sequence + "\n",
			text=True,
			capture_output=True,
			check=True,
		)
	except subprocess.CalledProcessError as exc:
		stderr = exc.stderr.strip() if exc.stderr else ""
		raise RuntimeError(
			f"RNAfold failed for sequence starting {exon_sequence[:20]!r}. {stderr}".strip()
		) from exc

	for line in result.stdout.splitlines():
		match = FEATURE_LINE_RE.search(line.strip())
		if match is not None:
			return float(match.group(1)), float(match.group(2))

	raise RuntimeError(
		"Could not parse RNAfold output for frequency/diversity. "
		f"Sequence starts with {exon_sequence[:20]!r}."
	)


def main() -> None:
	args = parse_args()
	check_rnafold_available(args.rnafold_bin)

	df = pd.read_csv(args.input_csv)
	if "exon" not in df.columns:
		raise ValueError(f"Input CSV {args.input_csv} must contain an 'exon' column.")

	freq_mfe_values: list[float] = []
	ensemble_diversity_values: list[float] = []

	total_rows = len(df)
	for idx, exon in enumerate(df["exon"].astype(str), start=1):
		freq_mfe, ensemble_diversity = compute_rnafold_features(exon, args.rnafold_bin)
		freq_mfe_values.append(freq_mfe)
		ensemble_diversity_values.append(ensemble_diversity)

		if idx % 100 == 0 or idx == total_rows:
			print(f"Processed {idx}/{total_rows} exons")

	df["freq_MFE"] = freq_mfe_values
	df["ensemble_diversity"] = ensemble_diversity_values

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output_csv, index=False)
	print(f"Saved dataset with RNA structure features to: {args.output_csv}")


if __name__ == "__main__":
	main()
