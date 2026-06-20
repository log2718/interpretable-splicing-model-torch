"""
Three figures for filter/weight visualization:

Fig 1 — Bottleneck weights: 4 subplots (incl_seq, incl_struct, skip_seq, skip_struct)
Fig 2 — Sequence filter logos: conv_incl and conv_skip (20 filters each, kernel=6)
Fig 3 — Structure filter heatmaps: conv_struct_incl and conv_struct_skip (8 filters each, kernel=30)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import logomaker
import pandas as pd
import sys

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
from model import PNASModel

OUT  = Path(__file__).resolve().parent
CKPT = BASE / "checkpoints/flank_150_30_uncertainty/best_model_20260613_171626.pt"

# ── Load weights ───────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd   = ckpt["model_state_dict"]

w_bottleneck = sd["variance_bottleneck.weight"].squeeze().numpy()  # (56,)

w_seq_incl    = sd["conv_incl.weight"].detach().numpy()             # (20, 4, 6)
w_seq_skip    = sd["conv_skip.weight"].detach().numpy()             # (20, 4, 6)
w_struct_incl = sd["conv_struct_incl.weight"].detach().numpy()      # (8, 8, 30)
w_struct_skip = sd["conv_struct_skip.weight"].detach().numpy()      # (8, 8, 30)

NUCS         = ["A", "C", "G", "T"]
STRUCT_CHANS = ["A", "C", "G", "T", "unpaired (.)", "open pair (", "close pair )", "wobble (GU)"]

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Bottleneck weights: 4 subplots
# ══════════════════════════════════════════════════════════════════════════════

segments = [
    ("Inclusion seq  (filters 0–19)",   w_bottleneck[ 0:20], "#2166ac"),
    ("Inclusion struct  (filters 20–27)", w_bottleneck[20:28], "#4dac26"),
    ("Skipping seq  (filters 28–47)",   w_bottleneck[28:48], "#d6604d"),
    ("Skipping struct  (filters 48–55)", w_bottleneck[48:56], "#f4a582"),
]

fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, (title, weights, color) in zip(axes1, segments):
    x = np.arange(len(weights))
    colors = [color if v >= 0 else "#888888" for v in weights]
    ax.bar(x, weights, color=colors, width=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Filter index", fontsize=8)
    ax.set_ylabel("Bottleneck weight", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7)

fig1.suptitle("Variance bottleneck weights  Linear(56→1)", fontsize=11)
fig1.tight_layout()
p1 = OUT / "bottleneck_weights_4panel.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p1}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Sequence filter logos (softmax over ACGT at each position)
# ══════════════════════════════════════════════════════════════════════════════

def softmax(x, axis=0):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def plot_seq_logos(weights, title, out_path):
    """weights: (N_filters, 4, kernel) — plot N_filters logos in a grid."""
    N = weights.shape[0]
    ncols = 5
    nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 1.8))
    axes = axes.flatten()

    for i in range(N):
        raw   = weights[i]                    # (4, 6)
        probs = softmax(raw, axis=0).T        # (6, 4) — softmax over ACGT at each pos
        df    = pd.DataFrame(probs, columns=NUCS)
        logomaker.Logo(df, ax=axes[i],
                       color_scheme="classic",
                       vpad=0.05, width=0.9)
        axes[i].set_title(f"Filter {i}", fontsize=7)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Hide unused axes
    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

plot_seq_logos(w_seq_incl, "Inclusion sequence filters  conv_incl  (20 × 4 × 6)",
               OUT / "seq_logos_incl.png")
plot_seq_logos(w_seq_skip, "Skipping sequence filters  conv_skip  (20 × 4 × 6)",
               OUT / "seq_logos_skip.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Structure filter heatmaps (raw weights, 8 channels × 30 positions)
# ══════════════════════════════════════════════════════════════════════════════

def plot_struct_heatmaps(weights, title, out_path):
    """weights: (N_filters, 8, 30) — plot each as an 8×30 heatmap."""
    N = weights.shape[0]
    ncols = 4
    nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 2.5))
    axes = axes.flatten()

    vmax = np.abs(weights).max()
    for i in range(N):
        ax  = axes[i]
        mat = weights[i]   # (8, 30)
        im  = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(8))
        ax.set_yticklabels(STRUCT_CHANS, fontsize=7)
        ax.set_xlabel("Position", fontsize=7)
        ax.set_title(f"Filter {i}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

plot_struct_heatmaps(w_struct_incl,
                     "Inclusion structure filters  conv_struct_incl  (8 × 8 × 30)",
                     OUT / "struct_heatmaps_incl.png")
plot_struct_heatmaps(w_struct_skip,
                     "Skipping structure filters  conv_struct_skip  (8 × 8 × 30)",
                     OUT / "struct_heatmaps_skip.png")
