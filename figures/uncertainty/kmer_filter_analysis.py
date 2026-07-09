"""
6-mer filter activation analysis for sequence filters.

Tests all 4^6 = 4096 possible 6-mers through conv_incl and conv_skip
(no position bias — pure kernel response) then produces:

Fig 1 — Activation histograms: distribution of Softplus(conv(6mer)) across
         all 4096 6-mers, one panel per filter (4×5 grid, incl and skip).
         Shows sparsity: narrow = specific motif, broad = nonspecific.

Fig 2 — Empirical PWM logos: activation-weighted nucleotide frequencies
         across all 4096 6-mers → sequence logo per filter (incl and skip).
         Compare with conv-weight logos in filter_plots.py as a sanity check.

Fig 3 — Heatmap: for each filter show its top-20 activated 6-mers and
         their activation values (filters × top kmers).
"""

from pathlib import Path
import itertools
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logomaker
import sys

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
from model import PNASModel

OUT  = Path(__file__).resolve().parent
CKPT = BASE / "checkpoints/flank_150_30_uncertainty/best_model_20260624_193658.pt"

BASES = list("ACGT")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PNASModel(input_length=250, use_batchnorm=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── Generate all 4096 6-mers and one-hot encode ───────────────────────────────
kmers = ["".join(k) for k in itertools.product(BASES, repeat=6)]  # 4096

def one_hot(seq: str) -> np.ndarray:
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        oh[BASES.index(b), i] = 1.0
    return oh

x = torch.tensor(np.stack([one_hot(k) for k in kmers]))  # (4096, 4, 6)
oh_array = x.numpy()                                       # (4096, 4, 6)

# ── Forward through conv only (no position bias) ──────────────────────────────
with torch.no_grad():
    act_incl = F.softplus(model.conv_incl(x)).squeeze(-1).numpy()  # (4096, 20)
    act_skip = F.softplus(model.conv_skip(x)).squeeze(-1).numpy()  # (4096, 20)

print(f"act_incl: shape={act_incl.shape}  mean={act_incl.mean():.4f}  "
      f"max={act_incl.max():.4f}")
print(f"act_skip: shape={act_skip.shape}  mean={act_skip.mean():.4f}  "
      f"max={act_skip.max():.4f}")

N_FILTERS = 20

# ── Fig 1: Activation histograms ──────────────────────────────────────────────

for label, act in [("incl", act_incl), ("skip", act_skip)]:
    fig, axes = plt.subplots(4, 5, figsize=(14, 10))
    for fi, ax in enumerate(axes.flat):
        vals = act[:, fi]
        ax.hist(vals, bins=50, color="#4393c3" if label == "incl" else "#d6604d",
                alpha=0.75, edgecolor="none")
        frac_zero = (vals < 0.01).mean()
        ax.set_title(f"Filter {fi}  (zero={frac_zero:.0%})", fontsize=8)
        ax.set_xlabel("Activation", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"6-mer activation distributions — seq filters ({label})\n"
                 f"(zero = fraction of 6-mers with activation < 0.01)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    p = OUT / f"kmer_activation_hist_{label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")

# ── Fig 2: Empirical PWM logos ────────────────────────────────────────────────
# For each filter i, position l, nucleotide j:
#   pwm[i,j,l] = sum_k( act[k,i] * oh[k,j,l] ) / sum_k( act[k,i] )
# = activation-weighted average nucleotide frequency at each position.

def activation_weighted_pwm(act: np.ndarray) -> np.ndarray:
    # Einsum indices:
    #   k = 4096 kmers
    #   i = 20 filters
    #   j = 4 nucleotides (ACGT)
    #   l = 6 positions
    # act:      (k=4096, i=20)
    # oh_array: (k=4096, j=4,   l=6)
    # output:   (i=20,   j=4,   l=6)
    weights = act + 1e-9
    pwm = np.einsum('ki,kjl->ijl', weights, oh_array) / weights.sum(0)[:, None, None]
    return pwm  # (N_FILTERS=20, 4, 6)

def pwm_to_ic(pwm: np.ndarray) -> np.ndarray:
    # pwm: (4, 6) frequencies; returns (6, 4) info-content matrix for logomaker
    pwm = np.clip(pwm, 1e-9, 1)
    ic_per_pos = (pwm * np.log2(pwm / 0.25)).sum(axis=0)  # (6,)
    return (pwm * ic_per_pos[None, :]).T  # (6, 4)

LOGO_OPTS = dict(color_scheme="classic", center_values=True,
                 flip_below=True, vpad=0.05, width=0.9)

for label, act in [("incl", act_incl), ("skip", act_skip)]:
    pwm_all = activation_weighted_pwm(act)  # (20, 4, 6)
    n_cols  = 5
    n_rows  = math.ceil(N_FILTERS / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 1.8))
    for fi, ax in enumerate(axes.flat):
        if fi >= N_FILTERS:
            ax.axis("off"); continue
        freq_df = pd.DataFrame(pwm_all[fi].T, columns=BASES)  # (6, 4) raw frequencies
        try:
            logomaker.Logo(freq_df, ax=ax, **LOGO_OPTS)
        except Exception:
            ax.axis("off")
        ax.set_title(f"Filter {fi}", fontsize=8)
        ax.set_xticks(range(6))
        ax.set_xticklabels(range(1, 7), fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle(f"Empirical PWM logos from 6-mer activations — seq filters ({label})\n"
                 f"(activation-weighted nucleotide frequencies, raw)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    p = OUT / f"kmer_pwm_logos_{label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")

# ── Fig 3: PWM heatmap (4×6 frequency matrix per filter) ─────────────────────
# Shows the same info as the logos but as a colour-coded grid.
# Each panel: rows = A C G T, cols = positions 1-6, colour = frequency (0→1).

for label, act in [("incl", act_incl), ("skip", act_skip)]:
    pwm_all = activation_weighted_pwm(act)  # (20, 4, 6)
    n_cols  = 5
    n_rows  = math.ceil(N_FILTERS / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 1.6))
    cmap = "Blues" if label == "incl" else "Reds"
    for fi, ax in enumerate(axes.flat):
        if fi >= N_FILTERS:
            ax.axis("off"); continue
        mat = pwm_all[fi]          # (4, 6)  rows=ACGT  cols=positions
        im  = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=0.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(range(1, 7), fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(BASES, fontsize=7)
        ax.set_title(f"Filter {fi}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02).ax.tick_params(labelsize=5)
    fig.suptitle(f"PWM heatmap — seq filters ({label})\n"
                 f"(activation-weighted nucleotide frequencies; uniform background = 0.25)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    p = OUT / f"kmer_pwm_heatmap_{label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")

# ── Fig 4: Heatmap of top-20 activated 6-mers per filter ─────────────────────

TOP_K = 20

for label, act in [("incl", act_incl), ("skip", act_skip)]:
    # For each filter collect top-K kmer indices
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    for fi, ax in enumerate(axes.flat):
        if fi >= N_FILTERS:
            ax.axis("off"); continue
        top_idx   = np.argsort(act[:, fi])[-TOP_K:][::-1]
        top_seqs  = [kmers[i] for i in top_idx]
        top_vals  = act[top_idx, fi]
        im = ax.imshow(top_vals[:, None], aspect="auto",
                       cmap="Blues" if label == "incl" else "Reds",
                       vmin=0)
        ax.set_yticks(range(TOP_K))
        ax.set_yticklabels(top_seqs, fontsize=6, fontfamily="monospace")
        ax.set_xticks([])
        ax.set_title(f"Filter {fi}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle(f"Top-{TOP_K} activated 6-mers per filter — seq filters ({label})",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    p = OUT / f"kmer_top_heatmap_{label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")

print("\nDone.")
