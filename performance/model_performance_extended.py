from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "test_data_rna_structure.csv"
OUTPUT_DIR = BASE_DIR / "performance" / "generated_files"
OUTPUT_PATH = OUTPUT_DIR / "metrics_violin_plot.png"
NUM_BINS = 20
FEATURE_LABELS = {
	"predicted_mfe": "MFE",
	"freq_MFE": "Frequency of MFE",
	"ensemble_diversity": "Ensemble Diversity",
}


def make_bin_series(df: pd.DataFrame, feature: str, bins: int = NUM_BINS) -> pd.Series:
	# qcut gives equal-count bins; duplicates="drop" handles repeated values safely.
	return pd.qcut(df[feature], q=bins, labels=False, duplicates="drop")


def kl_values_per_bin(
	df: pd.DataFrame,
	feature: str,
	bins: int = NUM_BINS,
	value_col: str = "kl",
) -> list[np.ndarray]:
	binned = make_bin_series(df, feature=feature, bins=bins)
	return [
		df.loc[binned == bin_index, value_col].to_numpy(dtype=float)
		for bin_index in range(bins)
	]


def plot_feature(
	ax: plt.Axes,
	bin_values: list[np.ndarray],
	feature: str,
	bins: int = NUM_BINS,
	value_label: str = "KL",
	percentiles: list[int] | None = None,
) -> None:
	positions = [index + 1 for index, values in enumerate(bin_values) if values.size > 0]
	data = [values for values in bin_values if values.size > 0]
	feature_label = FEATURE_LABELS.get(feature, feature)

	parts = ax.violinplot(
		data,
		positions=positions,
		showmeans=False,
		showmedians=True,
		showextrema=False,
	)
	for body in parts["bodies"]:
		body.set_facecolor("#4c78a8")
		body.set_edgecolor("#2f4b7c")
		body.set_alpha(0.75)

	ax.set_title(
		f"{value_label} distribution by {feature_label} bin",
		fontsize=12,
		pad=10,
	)
	ax.set_xlabel(f"{feature_label} bin", fontsize=11)
	ax.set_ylabel(value_label, fontsize=11)
	ax.grid(True, alpha=0.25)
	ax.set_xticks(range(1, bins + 1))
	ax.set_xticklabels([str(index) for index in range(1, bins + 1)])
	ax.tick_params(axis="x", labelbottom=True)
	all_values = np.concatenate([values for values in bin_values if values.size > 0])
	if all_values.size == 0:
		ax.set_ylim(0.0, 1.0)
	else:
		ymin = np.nanmin(all_values)
		ymax = np.nanmax(all_values)
		if not (np.isfinite(ymin) and np.isfinite(ymax)):
			ax.set_ylim(0.0, 1.0)
		else:
			span = ymax - ymin
			if span <= 0:
				pad = max(abs(ymax) * 0.1, 1e-6)
			else:
				pad = span * 0.05
			if ymin >= 0:
				lower = 0.0
			else:
				lower = ymin - pad
			upper = ymax + pad
			ax.set_ylim(lower, upper)

	# Draw percentile horizontal bars if requested
	if percentiles and len(data) > 0:
		# choose colors/styles for percentiles
		style_map = {}
		# default palette for percentiles
		palette = ["#e45756", "#4c78a8", "#54a24b", "#b279a2", "#ffb400"]
		for i, p in enumerate(percentiles):
			style_map[p] = {"color": palette[i % len(palette)], "linestyle": ["-", "--", ":", "-."][i % 4]}

		half_width = 0.18
		for idx, vals in enumerate(bin_values, start=1):
			if vals.size == 0:
				continue
			for p in percentiles:
				qv = np.percentile(vals, p)
				ax.hlines(qv, idx - half_width, idx + half_width, linewidth=2.0, color=style_map[p]["color"], linestyles=style_map[p]["linestyle"])

		# add a concise legend (one entry per percentile)
		handles = [Line2D([0], [0], color=style_map[p]["color"], linestyle=style_map[p]["linestyle"], linewidth=2.0) for p in percentiles]
		labels = [f"{p}th pct" for p in percentiles]
		ax.legend(handles, labels, loc="upper right", fontsize=9)


def main() -> None:
	df = pd.read_csv(CSV_PATH)
	required_cols = {"kl", "predicted_mfe", "freq_MFE", "ensemble_diversity"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in {CSV_PATH}: {sorted(missing)}")

	plot_specs: list[str] = ["predicted_mfe", "freq_MFE", "ensemble_diversity"]


	# First: standard KL violin figure
	plt.close("all")
	fig, axes = plt.subplots(len(plot_specs), 1, figsize=(13, 18), sharex=False)
	for ax, feature in zip(axes, plot_specs):
		bin_values = kl_values_per_bin(df, feature=feature, bins=NUM_BINS, value_col="kl")
		plot_feature(ax=ax, bin_values=bin_values, feature=feature, bins=NUM_BINS, value_label="KL")

	fig.suptitle("KL distributions by feature bins (20 quantile bins)", fontsize=16, y=1.0)
	plt.tight_layout()

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
	plt.show()
	print(f"Saved figure to: {OUTPUT_PATH}")

	# Second: log-scaled KL violin figure
	df["log_kl"] = np.log10(df["kl"] + 1e-4)
	OUTPUT_PATH_LOG = OUTPUT_DIR / "metrics_violin_plot_logkl.png"

	plt.close("all")
	fig, axes = plt.subplots(len(plot_specs), 1, figsize=(13, 18), sharex=False)
	for ax, feature in zip(axes, plot_specs):
		bin_values = kl_values_per_bin(df, feature=feature, bins=NUM_BINS, value_col="log_kl")
		plot_feature(
			ax=ax,
			bin_values=bin_values,
			feature=feature,
			bins=NUM_BINS,
			value_label="log10(KL + 1e-4)",
			percentiles=[60, 70, 80, 90, 95],
		)

	fig.suptitle("Log10-KL distributions by feature bins (20 quantile bins)", fontsize=16, y=1.0)
	plt.tight_layout()

	fig.savefig(OUTPUT_PATH_LOG, dpi=300, bbox_inches="tight")
	plt.show()
	print(f"Saved figure to: {OUTPUT_PATH_LOG}")


if __name__ == "__main__":
	main()
