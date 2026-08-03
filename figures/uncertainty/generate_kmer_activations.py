"""
Generate act_incl.npy, act_skip.npy, oh_array.npy for kmer_filter_analysis.py.

Run with anaconda python (needs torch):
  python figures/uncertainty/generate_kmer_activations.py \
      --weights data/model_weights_uncertainty.pt \
      --out-dir figures/uncertainty

  python figures/uncertainty/generate_kmer_activations.py \
      --weights data/model_weights.pt \
      --out-dir figures/uncertainty/original

Each 6-mer is embedded in the centre of a zero-padded 140-nt input and passed
through conv_incl / conv_skip (no position bias). Softplus is applied, then the
response is max-pooled over positions to give one scalar per (kmer, filter).
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
from model import PNASModel

BASES    = list("ACGT")
BASE_IDX = {b: i for i, b in enumerate(BASES)}
KMER_LEN = 6


def build_inputs(kmers: list[str]) -> tuple[torch.Tensor, np.ndarray]:
    """Return (seq_tensor, oh_array), both shape (N, 4, 6)."""
    N  = len(kmers)
    oh = np.zeros((N, 4, KMER_LEN), dtype=np.float32)
    for i, kmer in enumerate(kmers):
        for j, b in enumerate(kmer):
            oh[i, BASE_IDX[b], j] = 1.0
    return torch.from_numpy(oh), oh


@torch.no_grad()
def compute_activations(model, seq_tensor: torch.Tensor,
                        device: torch.device, batch_size: int = 512):
    """Run conv_incl / conv_skip + Softplus on (N, 4, 6) input.

    Conv output is (B, F, 1) — squeeze to (B, F).
    Returns act_incl, act_skip each of shape (N, n_filters).
    """
    N = seq_tensor.shape[0]
    act_incl_list, act_skip_list = [], []

    for start in range(0, N, batch_size):
        x = seq_tensor[start:start + batch_size].to(device)
        incl = F.softplus(model.conv_incl(x)).squeeze(2)  # (B, F)
        skip = F.softplus(model.conv_skip(x)).squeeze(2)  # (B, F)
        act_incl_list.append(incl.cpu().numpy())
        act_skip_list.append(skip.cpu().numpy())

    return np.concatenate(act_incl_list), np.concatenate(act_skip_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to .pt model weights file")
    parser.add_argument("--out-dir", default="figures/uncertainty",
                        help="Directory to save the .npy files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load only conv_incl / conv_skip weights directly — avoids position-bias
    # resampling issues when loading older checkpoints into the current model.
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    n_filters = state["conv_incl.weight"].shape[0]
    model = torch.nn.Module()
    model.conv_incl = torch.nn.Conv1d(4, n_filters, kernel_size=KMER_LEN, bias=True)
    model.conv_skip = torch.nn.Conv1d(4, n_filters, kernel_size=KMER_LEN, bias=True)
    model.conv_incl.weight.data.copy_(state["conv_incl.weight"])
    model.conv_incl.bias.data.copy_(state["conv_incl.bias"])
    model.conv_skip.weight.data.copy_(state["conv_skip.weight"])
    model.conv_skip.bias.data.copy_(state["conv_skip.bias"])
    model.to(device).eval()
    print(f"Loaded conv_incl/conv_skip from {args.weights}  (n_filters={n_filters})")

    # Build 4096 k-mers
    kmers = ["".join(k) for k in itertools.product(BASES, repeat=KMER_LEN)]
    print(f"Built {len(kmers)} {KMER_LEN}-mers")

    seq_tensor, oh_array = build_inputs(kmers)
    # oh_array: (N, 4, 6) rows=ACGT cols=positions — matches kmer_filter_analysis convention

    act_incl, act_skip = compute_activations(model, seq_tensor, device)
    print(f"act_incl: {act_incl.shape}  max={act_incl.max():.4f}")
    print(f"act_skip: {act_skip.shape}  max={act_skip.max():.4f}")

    np.save(out_dir / "act_incl.npy", act_incl)
    np.save(out_dir / "act_skip.npy", act_skip)
    np.save(out_dir / "oh_array.npy", oh_array)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
