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

def plot_seq_logos(weights, title, out_path):
    """weights: (N_filters, 4, kernel) — plot N_filters logos in a grid.
    Uses raw weights with center_values + flip_below so magnitude is preserved.
    """
    N = weights.shape[0]
    ncols = 5
    nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 1.8))
    axes = axes.flatten()

    for i in range(N):
        raw = weights[i]                      # (4, 6)
        df  = pd.DataFrame(raw.T, columns=NUCS)   # (6, 4)
        logomaker.Logo(df, ax=axes[i],
                       color_scheme="classic",
                       center_values=True,
                       flip_below=True,
                       vpad=0.05, width=0.9)
        axes[i].set_title(f"Filter {i}", fontsize=7)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

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

def _draw_struct_filter(ax_heat, ax_logo, kernel, vmax):
    """Draw heatmap + ACGT logo for one structure filter."""
    im = ax_heat.imshow(kernel, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(range(8))
    ax_heat.set_yticklabels(STRUCT_CHANS, fontsize=6)
    ax_heat.set_xlabel("Position", fontsize=7)
    plt.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.03)

    df = pd.DataFrame(kernel[:4, :].T, columns=NUCS)   # ACGT rows only
    logomaker.Logo(df, ax=ax_logo, color_scheme="classic",
                   center_values=True, flip_below=True, vpad=0.05, width=0.9)
    ax_logo.set_xticks([])
    ax_logo.set_yticks([])


def plot_struct_heatmaps(weights, title, out_path):
    """weights: (N_filters, 8, 30) — heatmap + ACGT logo per filter."""
    N     = weights.shape[0]   # 8
    ncols = 4
    vmax  = np.abs(weights).max()

    # 2 filter rows of 4 each; each filter row = [heatmap row, logo row]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(ncols * 4.5, 9))
    gs  = GridSpec(4, ncols, figure=fig,
                   height_ratios=[3, 1.2, 3, 1.2], hspace=0.55, wspace=0.4)

    for i in range(N):
        grp = i // ncols   # 0 or 1
        col = i % ncols
        ax_heat = fig.add_subplot(gs[grp * 2,     col])
        ax_logo = fig.add_subplot(gs[grp * 2 + 1, col])
        ax_heat.set_title(f"Filter {i}", fontsize=8)
        _draw_struct_filter(ax_heat, ax_logo, weights[i], vmax)

    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

plot_struct_heatmaps(w_struct_incl,
                     "Inclusion structure filters  conv_struct_incl  (8 × 8 × 30)",
                     OUT / "struct_heatmaps_incl.png")
plot_struct_heatmaps(w_struct_skip,
                     "Skipping structure filters  conv_struct_skip  (8 × 8 × 30)",
                     OUT / "struct_heatmaps_skip.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Combined: filters with |bottleneck weight| > 0.05, sorted descending
# ══════════════════════════════════════════════════════════════════════════════

THRESH = 0.05

def _collect(w_bn_slice, kernels, ftype):
    """Return list of (weight, ftype, local_idx, kernel) with |w|>THRESH, sorted by |w| desc."""
    entries = []
    for i, wt in enumerate(w_bn_slice):
        if abs(wt) > THRESH:
            entries.append((wt, ftype, i, kernels[i]))
    entries.sort(key=lambda x: -abs(x[0]))
    return entries

seq_incl_entries    = _collect(w_bottleneck[ 0:20], w_seq_incl,    "seq")
seq_skip_entries    = _collect(w_bottleneck[28:48], w_seq_skip,    "seq")
struct_incl_entries = _collect(w_bottleneck[20:28], w_struct_incl, "struct")
struct_skip_entries = _collect(w_bottleneck[48:56], w_struct_skip, "struct")

sections = [
    ("Sequence filters — inclusion",  seq_incl_entries),
    ("Sequence filters — skipping",   seq_skip_entries),
    ("Structure filters — inclusion", struct_incl_entries),
    ("Structure filters — skipping",  struct_skip_entries),
]

for title, entries in sections:
    print(f"{title}: {len(entries)} filters")

COL_RATIOS  = [0.6, 1.5, 4]
vmax_struct = max(np.abs(w_struct_incl).max(), np.abs(w_struct_skip).max())

# Build flat row list across 4 sections
# Each item: (height, row_type, payload)
#   row_type: "title" | "seq" | "struct_logo" | "struct_heat" | "sep"
all_rows = []
for sec_idx, (sec_title, entries) in enumerate(sections):
    if sec_idx > 0:
        all_rows.append((0.35, "sep",        sec_title))
    all_rows.append(    (0.5,  "title",      sec_title))
    for e in entries:
        wt, ftype, fidx, kernel = e
        if ftype == "seq":
            all_rows.append((3.0,  "seq",         e))
        else:
            all_rows.append((1.3,  "struct_logo",  e))
            all_rows.append((2.8,  "struct_heat",  e))

heights = [r[0] for r in all_rows]
total_h = sum(heights) * 0.72 + 1.0

fig4 = plt.figure(figsize=(14, total_h))
gs4  = fig4.add_gridspec(len(all_rows), 3, height_ratios=heights,
                         width_ratios=COL_RATIOS, hspace=0.12, wspace=0.25)

for ri, (_, rtype, payload) in enumerate(all_rows):

    if rtype == "sep":
        ax = fig4.add_subplot(gs4[ri, :])
        ax.axhline(0.5, color="#cccccc", linewidth=1)
        ax.axis("off")
        continue

    if rtype == "title":
        ax = fig4.add_subplot(gs4[ri, :])
        ax.text(0.01, 0.5, payload, ha="left", va="center",
                fontsize=10, fontweight="bold", color="#222222",
                transform=ax.transAxes)
        ax.axis("off")
        continue

    wt, ftype, fidx, kernel = payload

    if rtype == "seq":
        ax_lbl  = fig4.add_subplot(gs4[ri, 0])
        ax_logo = fig4.add_subplot(gs4[ri, 1])
        ax_empty = fig4.add_subplot(gs4[ri, 2])
        ax_lbl.axis("off"); ax_empty.axis("off")
        ax_lbl.text(0.5, 0.5, f"filter {fidx}\nw={wt:+.3f}",
                    ha="center", va="center", fontsize=8,
                    transform=ax_lbl.transAxes)
        df = pd.DataFrame(kernel.T, columns=NUCS)
        logomaker.Logo(df, ax=ax_logo, color_scheme="classic",
                       center_values=True, flip_below=True, vpad=0.05, width=0.9)
        ax_logo.set_xticks([]); ax_logo.set_yticks([])

    elif rtype == "struct_logo":
        ax_lbl  = fig4.add_subplot(gs4[ri, 0])
        ax_logo = fig4.add_subplot(gs4[ri, 1])
        ax_empty = fig4.add_subplot(gs4[ri, 2])
        ax_lbl.axis("off"); ax_empty.axis("off")
        ax_lbl.text(0.5, 0.5, f"filter {fidx}\nw={wt:+.3f}",
                    ha="center", va="center", fontsize=8,
                    transform=ax_lbl.transAxes)
        df = pd.DataFrame(kernel[:4, :].T, columns=NUCS)
        logomaker.Logo(df, ax=ax_logo, color_scheme="classic",
                       center_values=True, flip_below=True, vpad=0.05, width=0.9)
        ax_logo.set_xticks([]); ax_logo.set_yticks([])

    elif rtype == "struct_heat":
        ax_lbl   = fig4.add_subplot(gs4[ri, 0])
        ax_empty = fig4.add_subplot(gs4[ri, 1])
        ax_heat  = fig4.add_subplot(gs4[ri, 2])
        ax_lbl.axis("off"); ax_empty.axis("off")
        im = ax_heat.imshow(kernel, aspect="auto", cmap="RdBu_r",
                            vmin=-vmax_struct, vmax=vmax_struct)
        ax_heat.set_yticks(range(8))
        ax_heat.set_yticklabels(STRUCT_CHANS, fontsize=6)
        ax_heat.set_xlabel("Position", fontsize=7)
        plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02)

p4 = OUT / "combined_filtered_weights.png"
fig4.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {p4}")
