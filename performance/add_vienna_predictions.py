"""Add flank_40_30 model predictions and RNAfold ensemble features to a CSV.

Enriches an annotated CSV with:
  - predicted_PSI_flanks:  PSI from the flank_40_30 fine-tuned checkpoint
  - kl_flanks:             per-exon KL divergence  KL(PSI ∥ predicted_PSI_flanks)
  - freq_MFE:              frequency of MFE structure in Boltzmann ensemble
  - ensemble_diversity:    ensemble diversity from RNAfold partition function

Usage:
    python performance/add_vienna_predictions.py \\
      --csv-path data/test_flank_40_30.csv \\
      --test-npz data/test_flank_40_30.npz \\
      --checkpoint checkpoints/flank_40_30/best_model_20260525_030826.pt \\
      --temperature 37 \\
      --no-batchnorm
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model import PNASModel
from utils import LEFT_FLANK, RIGHT_FLANK

# ── Constants ─────────────────────────────────────────────────────────────────

EPS = 1e-6
FEATURE_LINE_RE = re.compile(
    r"frequency of mfe structure in ensemble\s+([0-9eE+\-.]+);\s+"
    r"ensemble diversity\s+([0-9eE+\-.]+)"
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add predicted_PSI_flanks, kl_flanks, freq_MFE, and ensemble_diversity "
            "columns to the flank_40_30 annotated CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-path", type=Path,
        default=BASE_DIR / "data" / "test_flank_40_30.csv",
        help="Annotated CSV to update in-place.",
    )
    parser.add_argument(
        "--test-npz", type=Path,
        default=BASE_DIR / "data" / "test_flank_40_30.npz",
        help="Test NPZ produced by prepare_dataset.py (for model-ready tensors).",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=BASE_DIR / "checkpoints" / "flank_40_30" / "best_model_20260525_030826.pt",
        help="Model checkpoint (.pt) from checkpoints/flank_40_30/.",
    )
    parser.add_argument(
        "--temperature", type=float, default=37.0,
        help="RNAfold temperature in Celsius for ensemble features.",
    )
    parser.add_argument(
        "--no-batchnorm", action="store_true",
        help="Disable BatchNorm in PNASModel (must match training config).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--rnafold-bin", type=str, default="RNAfold",
        help="RNAfold executable name or path.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (auto-detected if omitted).",
    )
    return parser.parse_args()


# ── Model inference ───────────────────────────────────────────────────────────

def predict_psi(
    npz_path: Path,
    checkpoint_path: Path,
    *,
    use_batchnorm: bool = True,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run batched model inference and return predicted PSI values.

    Args:
        npz_path: Path to the test NPZ file.
        checkpoint_path: Path to the model checkpoint.
        use_batchnorm: Whether the model uses batchnorm.
        batch_size: Inference batch size.
        device: Torch device.

    Returns:
        1-D NumPy array of predicted PSI values (sigmoid-transformed).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = np.load(npz_path)
    x_seq = torch.as_tensor(data["seq_oh"])
    x_struct = torch.as_tensor(data["struct_oh"])
    x_wobble = torch.as_tensor(data["wobbles"])
    n = len(x_seq)

    # Load model
    model = PNASModel(input_length=x_seq.shape[-1], use_batchnorm=use_batchnorm)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw.get("model_state_dict", raw)
    model.load_partial_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Batched inference
    predictions = []
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inference", unit="batch"):
            end = min(start + batch_size, n)
            seq = x_seq[start:end].to(device)
            struct = x_struct[start:end].to(device)
            wobble = x_wobble[start:end].to(device)

            logits = model(seq, struct, wobble, return_logits=True)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.append(probs)

    return np.concatenate(predictions)


# ── KL divergence ─────────────────────────────────────────────────────────────

def clip_probability(value: float, eps: float = EPS) -> float:
    """Clip a probability to (eps, 1-eps)."""
    return min(max(value, eps), 1.0 - eps)


def kl_divergence_binary(p: float, q: float) -> float:
    """Binary KL divergence: KL(Bernoulli(p) || Bernoulli(q))."""
    p = clip_probability(p)
    q = clip_probability(q)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


# ── RNAfold ensemble features ────────────────────────────────────────────────

def check_rnafold_available(rnafold_bin: str) -> None:
    """Raise FileNotFoundError if RNAfold is not on PATH."""
    if Path(rnafold_bin).expanduser().exists():
        return
    if shutil.which(rnafold_bin) is not None:
        return
    raise FileNotFoundError(
        f"Could not find RNAfold executable {rnafold_bin!r}. "
        "Install ViennaRNA and make RNAfold available on PATH, "
        "or pass --rnafold-bin with an absolute path."
    )


def compute_rnafold_features(
    exon_sequence: str,
    rnafold_bin: str,
    temperature: float,
) -> tuple[float, float]:
    """Run RNAfold -p to get freq_MFE and ensemble_diversity for one exon.

    Args:
        exon_sequence: Raw exon sequence (flanks are added internally).
        rnafold_bin: RNAfold executable.
        temperature: Folding temperature in Celsius.

    Returns:
        Tuple of (freq_MFE, ensemble_diversity).
    """
    full_sequence = (LEFT_FLANK + exon_sequence + RIGHT_FLANK).upper().replace("T", "U")

    try:
        result = subprocess.run(
            [rnafold_bin, "-p", "--noPS", "-T", str(temperature)],
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Validate inputs
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not args.test_npz.exists():
        raise FileNotFoundError(f"NPZ not found: {args.test_npz}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    check_rnafold_available(args.rnafold_bin)

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV: {args.csv_path}  ({len(df)} rows)")

    # ── Step 1: Model inference → predicted_PSI_flanks ────────────────────
    print("\n[1/3] Running model inference...")
    psi_preds = predict_psi(
        args.test_npz,
        args.checkpoint,
        use_batchnorm=not args.no_batchnorm,
        batch_size=args.batch_size,
        device=device,
    )

    if len(psi_preds) != len(df):
        raise ValueError(
            f"Row count mismatch: NPZ has {len(psi_preds)} examples, "
            f"CSV has {len(df)} rows."
        )

    df["predicted_PSI_flanks"] = psi_preds
    print(f"  predicted_PSI_flanks: mean={psi_preds.mean():.4f}, "
          f"std={psi_preds.std():.4f}")

    # ── Step 2: KL divergence → kl_flanks ────────────────────────────────
    print("\n[2/3] Computing KL divergence...")
    if "PSI" not in df.columns:
        raise ValueError("CSV is missing required 'PSI' column.")

    kl_values = [
        kl_divergence_binary(float(row["PSI"]), float(row["predicted_PSI_flanks"]))
        for _, row in df.iterrows()
    ]
    df["kl_flanks"] = kl_values
    print(f"  kl_flanks: mean={np.mean(kl_values):.6f}, "
          f"median={np.median(kl_values):.6f}")

    # ── Step 3: RNAfold ensemble features → freq_MFE, ensemble_diversity ─
    print(f"\n[3/3] Computing RNAfold ensemble features (T={args.temperature}°C)...")
    if "exon" not in df.columns:
        raise ValueError("CSV is missing required 'exon' column.")

    freq_mfe_values: list[float] = []
    ensemble_diversity_values: list[float] = []

    for exon in tqdm(df["exon"].astype(str), desc="RNAfold -p", unit="exon"):
        freq_mfe, ensemble_div = compute_rnafold_features(
            exon, args.rnafold_bin, args.temperature
        )
        freq_mfe_values.append(freq_mfe)
        ensemble_diversity_values.append(ensemble_div)

    df["freq_MFE"] = freq_mfe_values
    df["ensemble_diversity"] = ensemble_diversity_values
    print(f"  freq_MFE: mean={np.mean(freq_mfe_values):.6f}")
    print(f"  ensemble_diversity: mean={np.mean(ensemble_diversity_values):.6f}")

    # ── Write back in-place ──────────────────────────────────────────────
    df.to_csv(args.csv_path, index=False)
    print(f"\n✓ Updated CSV in-place: {args.csv_path}")
    print(f"  Columns added: predicted_PSI_flanks, kl_flanks, freq_MFE, ensemble_diversity")
    print(f"  Total columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
