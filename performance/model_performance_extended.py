from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "test_data_rna_structure.csv"
OUTPUT_DIR = BASE_DIR / "performance" / "generated_files"
OUTPUT_PATH = OUTPUT_DIR / "model_performance_large_KL.png"
NUM_BINS = 20
KL_THRESHOLDS = [0.5, 1.0]


def make_bin_series(df: pd.DataFrame, feature: str, bins: int = NUM_BINS) -> pd.Series:
	# qcut gives equal-count bins; duplicates="drop" handles repeated values safely.
	return pd.qcut(df[feature], q=bins, labels=False, duplicates="drop")


def proportion_above_threshold_per_bin(
	df: pd.DataFrame,
	feature: str,
	threshold: float,
	bins: int = NUM_BINS,
) -> pd.DataFrame:
	binned = make_bin_series(df, feature=feature, bins=bins)
	grouped = (
		df.assign(_bin=binned, is_above=df["kl"] > threshold)
		.groupby("_bin", observed=True)
		.agg(proportion=("is_above", "mean"))
		.reset_index()
		.sort_values("_bin")
	)

	# Always expose bins 1..20 on x-axis. Missing bins appear as NaN.
	all_bins = pd.DataFrame({"_bin": np.arange(bins, dtype=int)})
	grouped = all_bins.merge(grouped, on="_bin", how="left")
	grouped["bin"] = grouped["_bin"] + 1
	return grouped


def plot_feature(
	ax: plt.Axes,
	summary: pd.DataFrame,
	feature: str,
	threshold: float,
	bins: int = NUM_BINS,
) -> None:
	x = summary["bin"].to_numpy(dtype=float)
	y = summary["proportion"].to_numpy(dtype=float)

	ax.plot(x, y, marker="o", linewidth=2.0, markersize=4)
	if feature == "predicted_mfe":
		feature_label = "MFE"
	elif feature == "freq_MFE":
		feature_label = "Frequency of MFE"
	elif feature == "ensemble_diversity":
		feature_label = "Ensemble Diversity"
	else:
		feature_label = feature

	ax.set_title(
		f"Proportion of KL > {threshold:g} by {feature_label}",
		fontsize=12,
		pad=10,
	)
	ax.set_xlabel(f"{feature_label} bin", fontsize=11)
	ax.set_ylabel("Proportion (KL > threshold)", fontsize=11)
	ax.grid(True, alpha=0.25)
	ax.set_xticks(range(1, bins + 1))
	if y.size == 0:
		ymax = 1.0
	else:
		ymax = np.nanmax(y)
	if not np.isfinite(ymax) or ymax <= 0:
		ymax = 1.0
	ax.set_ylim(0.0, ymax * 1.15)


def main() -> None:
	df = pd.read_csv(CSV_PATH)
	required_cols = {"kl", "predicted_mfe", "freq_MFE", "ensemble_diversity"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in {CSV_PATH}: {sorted(missing)}")

	plot_specs: list[tuple[str, float]] = []
	for threshold in KL_THRESHOLDS:
		plot_specs.append(("predicted_mfe", threshold))
		plot_specs.append(("freq_MFE", threshold))
		plot_specs.append(("ensemble_diversity", threshold))

	plt.close("all")
	fig, axes = plt.subplots(len(plot_specs), 1, figsize=(12, 24))
	for ax, (feature, threshold) in zip(axes, plot_specs):
		summary = proportion_above_threshold_per_bin(
			df,
			feature=feature,
			threshold=threshold,
			bins=NUM_BINS,
		)
		plot_feature(
			ax=ax,
			summary=summary,
			feature=feature,
			threshold=threshold,
			bins=NUM_BINS,
		)

	fig.suptitle(
		"Large-KL proportion by feature bins (20 quantile bins)",
		fontsize=16,
		y=1.0,
	)
	plt.tight_layout()

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
	plt.show()
	print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
