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
CKPT = BASE / "checkpoints/flank_150_30_uncertainty/best_model_20260624_193658.pt"

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

# ── Compute mean activations via forward hook ──────────────────────────────────
# contribution_i = w_i * mean(h_i)  — the actual average push of filter i on z

NPZ  = BASE / "data/test_flank_150_30.npz"
data = np.load(NPZ)

model = PNASModel(input_length=250, use_batchnorm=False)
model.load_state_dict(sd)
model.eval()

from torch.utils.data import DataLoader, TensorDataset

x_seq    = torch.tensor(data["seq_oh"],  dtype=torch.float32)
x_struct = torch.tensor(data["struct_oh"], dtype=torch.float32)
x_wob    = torch.tensor(data["wobbles"],   dtype=torch.float32)
loader   = DataLoader(TensorDataset(x_seq, x_struct, x_wob), batch_size=512)

h_all = []
hook  = model.variance_bottleneck.register_forward_hook(
    lambda m, inp, out: h_all.append(inp[0].detach().cpu())
)
with torch.no_grad():
    for seq, struct, wob in loader:
        model(seq, struct, wob, return_uncertainty=True)
hook.remove()

h_mean        = torch.cat(h_all).mean(dim=0).numpy()   # (56,) mean activation per filter
contributions = w_bottleneck * h_mean                   # actual contribution to z
print(f"h_mean range: [{h_mean.min():.4f}, {h_mean.max():.4f}]")
print(f"contributions range: [{contributions.min():.4f}, {contributions.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Bottleneck weights: 4 subplots
# ══════════════════════════════════════════════════════════════════════════════

segments = [
    ("Inclusion seq  (filters 0–19)",    contributions[ 0:20], w_bottleneck[ 0:20], "#2166ac"),
    ("Inclusion struct  (filters 20–27)", contributions[20:28], w_bottleneck[20:28], "#4dac26"),
    ("Skipping seq  (filters 28–47)",    contributions[28:48], w_bottleneck[28:48], "#d6604d"),
    ("Skipping struct  (filters 48–55)", contributions[48:56], w_bottleneck[48:56], "#f4a582"),
]

fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, (title, contribs, weights, color) in zip(axes1, segments):
    x = np.arange(len(contribs))
    colors = [color if v >= 0 else "#888888" for v in contribs]
    ax.bar(x, contribs, color=colors, width=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Filter index", fontsize=8)
    ax.set_ylabel("Contribution (w × mean activation)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7)

fig1.suptitle("Variance bottleneck contributions  (mean activation × weight)", fontsize=11)
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

THRESH = 0.01   # threshold on |contribution| not raw weight

def _collect(contrib_slice, w_slice, kernels, ftype):
    """Return list of (contribution, weight, ftype, local_idx, kernel)
    with |contribution|>THRESH, sorted by |contribution| desc."""
    entries = []
    for i, (ct, wt) in enumerate(zip(contrib_slice, w_slice)):
        if abs(ct) > THRESH:
            entries.append((ct, wt, ftype, i, kernels[i]))
    entries.sort(key=lambda x: -abs(x[0]))
    return entries

seq_incl_entries    = _collect(contributions[ 0:20], w_bottleneck[ 0:20], w_seq_incl,    "seq")
seq_skip_entries    = _collect(contributions[28:48], w_bottleneck[28:48], w_seq_skip,    "seq")
struct_incl_entries = _collect(contributions[20:28], w_bottleneck[20:28], w_struct_incl, "struct")
struct_skip_entries = _collect(contributions[48:56], w_bottleneck[48:56], w_struct_skip, "struct")

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


def _build_section_rows(entries):
    """Return flat row list for one section (no title row)."""
    rows = []
    for e in entries:
        ct, wt, ftype, fidx, kernel = e
        if ftype == "seq":
            rows.append((3.0, "seq",         e))
        else:
            rows.append((1.3, "struct_logo", e))
            rows.append((2.8, "struct_heat", e))
    return rows


def _draw_rows(fig, gs, rows):
    for ri, (_, rtype, payload) in enumerate(rows):
        ct, wt, ftype, fidx, kernel = payload
        lbl_text = f"filter {fidx}\ncontrib={ct:+.3f}\nw={wt:+.3f}"

        if rtype == "seq":
            ax_lbl   = fig.add_subplot(gs[ri, 0])
            ax_logo  = fig.add_subplot(gs[ri, 1])
            ax_empty = fig.add_subplot(gs[ri, 2])
            ax_lbl.axis("off"); ax_empty.axis("off")
            ax_lbl.text(0.5, 0.5, lbl_text,
                        ha="center", va="center", fontsize=7,
                        transform=ax_lbl.transAxes)
            df = pd.DataFrame(kernel.T, columns=NUCS)
            logomaker.Logo(df, ax=ax_logo, color_scheme="classic",
                           center_values=True, flip_below=True, vpad=0.05, width=0.9)
            ax_logo.set_xticks([]); ax_logo.set_yticks([])

        elif rtype == "struct_logo":
            ax_lbl   = fig.add_subplot(gs[ri, 0])
            ax_logo  = fig.add_subplot(gs[ri, 1])
            ax_empty = fig.add_subplot(gs[ri, 2])
            ax_lbl.axis("off"); ax_empty.axis("off")
            ax_lbl.text(0.5, 0.5, lbl_text,
                        ha="center", va="center", fontsize=7,
                        transform=ax_lbl.transAxes)
            df = pd.DataFrame(kernel[:4, :].T, columns=NUCS)
            logomaker.Logo(df, ax=ax_logo, color_scheme="classic",
                           center_values=True, flip_below=True, vpad=0.05, width=0.9)
            ax_logo.set_xticks([]); ax_logo.set_yticks([])

        elif rtype == "struct_heat":
            ax_lbl   = fig.add_subplot(gs[ri, 0])
            ax_empty = fig.add_subplot(gs[ri, 1])
            ax_heat  = fig.add_subplot(gs[ri, 2])
            ax_lbl.axis("off"); ax_empty.axis("off")
            im = ax_heat.imshow(kernel, aspect="auto", cmap="RdBu_r",
                                vmin=-vmax_struct, vmax=vmax_struct)
            ax_heat.set_yticks(range(8))
            ax_heat.set_yticklabels(STRUCT_CHANS, fontsize=6)
            ax_heat.set_xlabel("Position", fontsize=7)
            plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02)


def _save_seq_2col(sec_title, entries, out_path):
    """Seq filters in 2 columns: left = top half (higher contribution), right = lower half."""
    import math
    n_left  = math.ceil(len(entries) / 2)
    left    = entries[:n_left]
    right   = entries[n_left:]
    n_rows  = n_left   # left col is always >= right col

    ROW_H   = 3.0
    total_h = n_rows * ROW_H * 0.72 + 1.2
    # widths: [lbl_L, logo_L, gap, lbl_R, logo_R]
    fig = plt.figure(figsize=(12, total_h))
    gs  = fig.add_gridspec(n_rows, 5,
                           height_ratios=[ROW_H] * n_rows,
                           width_ratios=[0.5, 2.0, 0.2, 0.5, 2.0],
                           hspace=0.12, wspace=0.15)

    def _draw_seq(col_entries, lbl_col, logo_col):
        for ri, e in enumerate(col_entries):
            ct, wt, ftype, fidx, kernel = e
            ax_lbl  = fig.add_subplot(gs[ri, lbl_col])
            ax_logo = fig.add_subplot(gs[ri, logo_col])
            ax_lbl.axis("off")
            ax_lbl.text(0.5, 0.5, f"filter {fidx}\ncontrib={ct:+.3f}\nw={wt:+.3f}",
                        ha="center", va="center", fontsize=7,
                        transform=ax_lbl.transAxes)
            df = pd.DataFrame(kernel.T, columns=NUCS)
            logomaker.Logo(df, ax=ax_logo, color_scheme="classic",
                           center_values=True, flip_below=True, vpad=0.05, width=0.9)
            ax_logo.set_xticks([]); ax_logo.set_yticks([])

    _draw_seq(left,  lbl_col=0, logo_col=1)
    _draw_seq(right, lbl_col=3, logo_col=4)

    # blank gap col
    for ri in range(n_rows):
        ax = fig.add_subplot(gs[ri, 2]); ax.axis("off")

    fig.suptitle(sec_title, fontsize=12, fontweight="bold", y=1.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


SEC_SLUGS = [
    "seq_incl",
    "seq_skip",
    "struct_incl",
    "struct_skip",
]

for (sec_title, entries), slug in zip(sections, SEC_SLUGS):
    out_path = OUT / f"combined_{slug}.png"

    if slug.startswith("seq"):
        _save_seq_2col(sec_title, entries, out_path)
    else:
        rows    = _build_section_rows(entries)
        heights = [r[0] for r in rows]
        total_h = sum(heights) * 0.72 + 1.2

        fig = plt.figure(figsize=(14, total_h))
        gs  = fig.add_gridspec(len(rows), 3, height_ratios=heights,
                               width_ratios=COL_RATIOS, hspace=0.12, wspace=0.25)
        _draw_rows(fig, gs, rows)
        fig.suptitle(sec_title, fontsize=12, fontweight="bold", y=1.0)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")
