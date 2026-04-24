"""Add per-exon KL divergence to the RNA-structure test CSV in-place."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR / "data" / "test_data_rna_structure.csv"
EPS = 1e-6


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compute KL divergence per exon from PSI and predicted_PSI and "
			"store it back into the same CSV."
		)
	)
	parser.add_argument(
		"--csv-path",
		type=Path,
		default=DEFAULT_CSV,
		help="Path to RNA-structure CSV file.",
	)
	return parser.parse_args()


def clip_probability(value: float, eps: float = EPS) -> float:
	return min(max(value, eps), 1.0 - eps)


def kl_divergence_binary(p: float, q: float) -> float:
	p = clip_probability(p)
	q = clip_probability(q)
	return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def main() -> None:
	args = parse_args()
	csv_path = args.csv_path

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError(f"CSV has no header: {csv_path}")
		required = {"PSI", "predicted_PSI"}
		missing = required - set(reader.fieldnames)
		if missing:
			raise ValueError(
				f"CSV is missing required columns: {sorted(missing)} in {csv_path}"
			)
		rows = list(reader)
		fieldnames = list(reader.fieldnames)

	if "kl" not in fieldnames:
		fieldnames.append("kl")

	for row in rows:
		p = float(row["PSI"])
		q = float(row["predicted_PSI"])
		row["kl"] = f"{kl_divergence_binary(p, q):.16g}"

	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Updated KL values in: {csv_path}")
	print(f"Rows processed: {len(rows)}")


if __name__ == "__main__":
	main()

