"""Add or update per-config columns in data/test_annotated.csv.

Stages (all enabled by default; use flags to run only a subset):
  1. inference  → predicted_PSI_{config}
  2. kl         → kl_{config}
  3. rnafold    → MFE_{config}, freq_MFE_{config}, ens_div_{config}

Existing non-NaN columns are skipped unless --force is passed.

Example — re-run ensemble features for vienna60 at the correct temperature:
    python performance/enrich.py \\
      --csv data/test_annotated.csv \\
      --config-name vienna60 \\
      --temperature 60 \\
      --rnafold-only

Example — full enrichment for a new config:
    python performance/enrich.py \\
      --csv data/test_annotated.csv \\
      --config-name flank_100_30 \\
      --npz data/test_flank_100_30.npz \\
      --checkpoint checkpoints/flank_100_30/best_model_*.pt \\
      --temperature 37
"""

from __future__ import annotations

import argparse
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

from performance.lib.kl import kl_bernoulli_vec
from performance.lib.rnafold import check_rnafold, fold_ensemble, fold_gquad, fold_mfe


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add per-config columns to data/test_annotated.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",         type=Path, default=BASE_DIR / "data" / "test_annotated.csv")
    p.add_argument("--config-name", required=True, help="Config label, e.g. vienna60 or flank_100_30")
    p.add_argument("--npz",         type=Path, default=None, help="Test NPZ (required for inference stage)")
    p.add_argument("--checkpoint",  type=Path, default=None, help="Model .pt checkpoint (required for inference)")
    p.add_argument("--temperature", type=float, default=None, help="RNAfold temperature in °C (required for rnafold stage)")
    p.add_argument("--left-flank",  type=str,  default=None, help="Override LEFT_FLANK (default: utils.LEFT_FLANK)")
    p.add_argument("--right-flank", type=str,  default=None, help="Override RIGHT_FLANK (default: utils.RIGHT_FLANK)")
    p.add_argument("--no-batchnorm", action="store_true")
    p.add_argument("--batch-size",   type=int, default=256)
    p.add_argument("--rnafold-bin",  type=str, default="RNAfold")
    p.add_argument("--device",       type=str, default=None)
    p.add_argument("--force",        action="store_true", help="Overwrite existing non-NaN columns")

    # Stage selection
    p.add_argument(
        "--gquad", action="store_true",
        help="Also compute gquad_present_{config} and MFE_delta_gquad_{config} via RNAfold -g."
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument("--inference-only", action="store_true")
    g.add_argument("--kl-only",        action="store_true")
    g.add_argument("--rnafold-only",   action="store_true")

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _needs_fill(df: pd.DataFrame, col: str, force: bool) -> bool:
    """Return True if `col` should be computed (missing or force-overwrite)."""
    if col not in df.columns:
        return True
    if force:
        return True
    return df[col].isna().any()


def _predict_psi(
    npz_path: Path,
    checkpoint_path: Path,
    use_batchnorm: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    data     = np.load(npz_path)
    x_seq    = torch.as_tensor(data["seq_oh"])
    x_struct = torch.as_tensor(data["struct_oh"])
    x_wobble = torch.as_tensor(data["wobbles"])
    n        = len(x_seq)

    model = PNASModel(input_length=x_seq.shape[-1], use_batchnorm=use_batchnorm)
    raw   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_partial_state_dict(raw.get("model_state_dict", raw))
    model = model.to(device).eval()

    preds = []
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inference", unit="batch"):
            end = min(start + batch_size, n)
            logits = model(
                x_seq[start:end].to(device),
                x_struct[start:end].to(device),
                x_wobble[start:end].to(device),
                return_logits=True,
            )
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = args.config_name

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    left_flank  = args.left_flank  or LEFT_FLANK
    right_flank = args.right_flank or RIGHT_FLANK

    run_inference = not (args.kl_only or args.rnafold_only)
    run_kl        = not (args.inference_only or args.rnafold_only)
    run_rnafold   = not (args.inference_only or args.kl_only)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv} ({len(df):,} rows, {len(df.columns)} cols)")

    # ── Stage 1: inference ────────────────────────────────────────────────────
    psi_col = f"predicted_PSI_{cfg}"
    if run_inference and _needs_fill(df, psi_col, args.force):
        if args.npz is None or args.checkpoint is None:
            raise ValueError("--npz and --checkpoint are required for the inference stage.")
        print(f"\n[inference] Computing {psi_col} ...")
        preds = _predict_psi(
            args.npz, args.checkpoint,
            use_batchnorm=not args.no_batchnorm,
            batch_size=args.batch_size,
            device=device,
        )
        if len(preds) != len(df):
            raise ValueError(f"NPZ has {len(preds)} rows but CSV has {len(df)} rows.")
        df[psi_col] = preds
        print(f"  {psi_col}: mean={preds.mean():.4f}, std={preds.std():.4f}")
    elif run_inference:
        print(f"\n[inference] {psi_col} already present — skipping (use --force to overwrite)")

    # ── Stage 2: KL divergence ────────────────────────────────────────────────
    kl_col = f"kl_{cfg}"
    if run_kl and _needs_fill(df, kl_col, args.force):
        if psi_col not in df.columns or df[psi_col].isna().all():
            raise ValueError(f"Column {psi_col!r} is missing — run inference stage first.")
        if "PSI" not in df.columns:
            raise ValueError("CSV is missing required 'PSI' column.")
        print(f"\n[kl] Computing {kl_col} ...")
        df[kl_col] = kl_bernoulli_vec(df["PSI"].values, df[psi_col].values)
        print(f"  {kl_col}: mean={df[kl_col].mean():.6f}, median={df[kl_col].median():.6f}")
    elif run_kl:
        print(f"\n[kl] {kl_col} already present — skipping (use --force to overwrite)")

    # ── Stage 3: RNAfold ──────────────────────────────────────────────────────
    mfe_col  = f"MFE_{cfg}"
    freq_col = f"freq_MFE_{cfg}"
    div_col  = f"ens_div_{cfg}"

    rnafold_needed = run_rnafold and any(
        _needs_fill(df, col, args.force) for col in [mfe_col, freq_col, div_col]
    )

    if rnafold_needed:
        if args.temperature is None:
            raise ValueError("--temperature is required for the RNAfold stage.")
        if "exon" not in df.columns:
            raise ValueError("CSV is missing required 'exon' column.")
        check_rnafold(args.rnafold_bin)
        print(f"\n[rnafold] Computing {mfe_col}, {freq_col}, {div_col} at T={args.temperature}°C ...")

        mfe_vals, freq_vals, div_vals = [], [], []
        for exon in tqdm(df["exon"].astype(str), desc="RNAfold", unit="exon"):
            full_seq = left_flank + exon + right_flank
            _, mfe        = fold_mfe(full_seq, args.rnafold_bin, args.temperature)
            freq, ens_div = fold_ensemble(full_seq, args.rnafold_bin, args.temperature)
            mfe_vals.append(mfe)
            freq_vals.append(freq)
            div_vals.append(ens_div)

        if _needs_fill(df, mfe_col,  args.force): df[mfe_col]  = mfe_vals
        if _needs_fill(df, freq_col, args.force): df[freq_col] = freq_vals
        if _needs_fill(df, div_col,  args.force): df[div_col]  = div_vals
        print(f"  {mfe_col}: mean={np.mean(mfe_vals):.4f}")
        print(f"  {freq_col}: mean={np.mean(freq_vals):.6f}")
        print(f"  {div_col}: mean={np.mean(div_vals):.6f}")
    elif run_rnafold:
        print(f"\n[rnafold] {mfe_col}/{freq_col}/{div_col} already present — skipping (use --force)")

    # ── Stage 4: G-quadruplex features (optional) ─────────────────────────────
    gquad_col     = f"gquad_present_{cfg}"
    mfe_delta_col = f"MFE_delta_gquad_{cfg}"

    if args.gquad and run_rnafold and any(
        _needs_fill(df, col, args.force) for col in [gquad_col, mfe_delta_col]
    ):
        if args.temperature is None:
            raise ValueError("--temperature is required for the gquad stage.")
        if "exon" not in df.columns:
            raise ValueError("CSV is missing required 'exon' column.")
        check_rnafold(args.rnafold_bin)
        print(f"\n[gquad] Computing {gquad_col}, {mfe_delta_col} at T={args.temperature}°C ...")

        # We need the normal MFE for delta computation.
        mfe_ref = df[mfe_col].values if mfe_col in df.columns else None
        if mfe_ref is None or np.isnan(mfe_ref).any():
            raise ValueError(
                f"Column {mfe_col!r} must be present before computing gquad delta. "
                "Run the rnafold stage first."
            )

        gquad_present_vals, mfe_delta_vals = [], []
        for exon, mfe_normal in tqdm(
            zip(df["exon"].astype(str), mfe_ref), total=len(df), desc="RNAfold -g", unit="exon"
        ):
            full_seq = left_flank + exon + right_flank
            gquad_struct, gquad_mfe = fold_gquad(full_seq, args.rnafold_bin, args.temperature)
            gquad_present_vals.append('+' in gquad_struct)
            mfe_delta_vals.append(float(gquad_mfe) - float(mfe_normal))

        if _needs_fill(df, gquad_col,     args.force): df[gquad_col]     = gquad_present_vals
        if _needs_fill(df, mfe_delta_col, args.force): df[mfe_delta_col] = mfe_delta_vals
        n_gquad = sum(gquad_present_vals)
        print(f"  {gquad_col}: {n_gquad}/{len(df)} ({100*n_gquad/len(df):.1f}%) gquad-positive")
        print(f"  {mfe_delta_col}: mean={np.mean(mfe_delta_vals):.4f}")
    elif args.gquad and run_rnafold:
        print(f"\n[gquad] {gquad_col}/{mfe_delta_col} already present — skipping (use --force)")

    # ── Write back ────────────────────────────────────────────────────────────
    df.to_csv(args.csv, index=False)
    print(f"\nUpdated {args.csv} ({len(df.columns)} columns total)")


if __name__ == "__main__":
    main()
