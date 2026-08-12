"""
Generate three sets of visualisation for each monitored-model seed (42, 142, 242):

  1. IC-scaled sequence logos  (kmer_pwm_logos_incl / skip)
     • All 4096 6-mers run through conv_incl / conv_skip + Softplus
     • Threshold: activation > max_activation / 4   (was /2)
     • IC-scaled PWM from unweighted nucleotide frequencies of active k-mers

  2. Structural filter heatmaps  (struct_heatmaps_incl / skip)
     • Raw weights of conv_struct_incl / conv_struct_skip   shape (8, 8, 30)
     • 8 input channels: [A, C, G, T, unpaired, open, close, wobble]

  3. Activation histograms  (kmer_activation_hist_incl / skip)
     • Distribution of Softplus(conv(6mer)) across all 4096 6-mers per filter
"""

from pathlib import Path
import itertools
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logomaker

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))

SEEDS   = [42, 142, 242]
BASES   = list("ACGT")
KMER_LEN = 6
N_FILTERS = 20

STRUCT_CHANS = [
    "A", "C", "G", "T",
    "unpaired (.)", "open pair (", "close pair )", "wobble (GU)",
]

# ── K-mer helpers ─────────────────────────────────────────────────────────────

def _build_kmers():
    kmers  = ["".join(k) for k in itertools.product(BASES, repeat=KMER_LEN)]
    idx    = {b: i for i, b in enumerate(BASES)}
    oh     = np.zeros((len(kmers), 4, KMER_LEN), dtype=np.float32)
    for i, kmer in enumerate(kmers):
        for j, b in enumerate(kmer):
            oh[i, idx[b], j] = 1.0
    return kmers, oh


@torch.no_grad()
def _compute_kmer_activations(conv_incl, conv_skip, oh, batch=512):
    t  = torch.from_numpy(oh)
    ai, as_ = [], []
    for s in range(0, len(t), batch):
        x = t[s:s + batch]
        ai.append(F.softplus(conv_incl(x)).squeeze(2).cpu().numpy())
        as_.append(F.softplus(conv_skip(x)).squeeze(2).cpu().numpy())
    return np.concatenate(ai), np.concatenate(as_)


# ── Plot 1: IC-scaled logos ────────────────────────────────────────────────────

def _threshold_ic_pwm(act, oh, threshold_frac=0.25):
    """Return (N_FILTERS, 6, 4) IC-scaled matrix + per-filter stats."""
    ic_mats, stats = [], []
    for fi in range(act.shape[1]):
        vals      = act[:, fi]
        max_act   = vals.max()
        thresh    = max_act * threshold_frac
        mask      = vals > thresh
        n_active  = int(mask.sum())
        if n_active == 0:
            ic_mats.append(np.zeros((KMER_LEN, 4)))
            stats.append({"n_active": 0, "max_act": float(max_act)})
            continue
        freq      = oh[mask].mean(axis=0)              # (4, 6)
        freq_safe = np.clip(freq, 1e-9, 1)
        H         = -(freq_safe * np.log2(freq_safe)).sum(axis=0)  # (6,)
        R         = 2 - H
        ic_scaled = (freq * R[None, :]).T              # (6, 4)
        ic_mats.append(ic_scaled)
        stats.append({"n_active": n_active, "max_act": float(max_act), "freq": freq})
    return np.array(ic_mats), stats


def plot_logos(act, oh, label, out_path, threshold_frac=0.25):
    ic_mats, stats = _threshold_ic_pwm(act, oh, threshold_frac)
    n_cols = 5
    n_rows = math.ceil(N_FILTERS / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 1.8))
    for fi, ax in enumerate(axes.flat):
        if fi >= N_FILTERS:
            ax.axis("off"); continue
        s  = stats[fi]
        df = pd.DataFrame(ic_mats[fi], columns=BASES, index=range(1, KMER_LEN + 1))
        try:
            logomaker.Logo(df, ax=ax, color_scheme="classic", vpad=0.05, width=0.9)
        except Exception:
            ax.axis("off")
        ax.set_ylim(0, 2)
        ax.set_title(f"Filter {fi}  (n={s['n_active']})", fontsize=7)
        ax.set_xticks(range(KMER_LEN))
        ax.set_xticklabels(range(1, KMER_LEN + 1), fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
    long_label = "inclusion" if label == "incl" else "skipping"
    fig.suptitle(
        f"IC-scaled logos — {long_label} seq filters\n"
        f"(threshold: activation > max / 4)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path.name}")


# ── Plot 2: Structural filter heatmaps ────────────────────────────────────────

def _draw_struct_filter(ax_heat, kernel, vmax):
    im = ax_heat.imshow(kernel, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(range(8))
    ax_heat.set_yticklabels(STRUCT_CHANS, fontsize=6)
    ax_heat.set_xlabel("Position", fontsize=7)
    plt.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.03)


def plot_struct_heatmaps(weights, label, out_path):
    """weights: (8, 8, 30)"""
    N     = weights.shape[0]
    ncols = 4
    vmax  = np.abs(weights).max()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(ncols * 4.5, 9))
    gs  = GridSpec(2, ncols, figure=fig, hspace=0.55, wspace=0.4)
    for i in range(N):
        row = i // ncols
        col = i % ncols
        ax  = fig.add_subplot(gs[row, col])
        ax.set_title(f"Filter {i}", fontsize=8)
        _draw_struct_filter(ax, weights[i], vmax)
    long_label = "inclusion" if label == "incl" else "skipping"
    fig.suptitle(
        f"Structural filter weights — {long_label}  conv_struct_{label}  (8 × 8 × 30)\n"
        f"Channels: {', '.join(STRUCT_CHANS)}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path.name}")


# ── Plot 3: Activation histograms ─────────────────────────────────────────────

def plot_activation_hist(act, label, out_path):
    color = "#4393c3" if label == "incl" else "#d6604d"
    fig, axes = plt.subplots(4, 5, figsize=(14, 10))
    for fi, ax in enumerate(axes.flat):
        vals = act[:, fi]
        ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor="none")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.set_xlim(0, act.max() * 1.05)
        frac_zero = (vals < 0.01).mean()
        ax.set_title(f"Filter {fi}  (zero={frac_zero:.0%})", fontsize=8)
        ax.set_xlabel("Activation", fontsize=7)
        ax.tick_params(labelsize=7)
    long_label = "inclusion" if label == "incl" else "skipping"
    fig.suptitle(f"6-mer activations (log y) — {long_label}", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path.name}")


# ── Main loop ─────────────────────────────────────────────────────────────────

import glob

kmers, oh_array = _build_kmers()

for seed in SEEDS:
    ckpt_paths = sorted(glob.glob(
        str(BASE / f"checkpoints/monitored_seed{seed}/stage5/best_model_*.pt")
    ))
    if not ckpt_paths:
        print(f"No stage5 checkpoint found for seed {seed}, skipping.")
        continue
    ckpt_path = ckpt_paths[-1]
    print(f"\n=== Seed {seed}  ({Path(ckpt_path).name}) ===")

    out_dir = Path(__file__).resolve().parent / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]

    # Lightweight conv-only modules for k-mer analysis
    conv_incl = torch.nn.Conv1d(4, N_FILTERS, kernel_size=KMER_LEN, bias=True)
    conv_skip = torch.nn.Conv1d(4, N_FILTERS, kernel_size=KMER_LEN, bias=True)
    conv_incl.weight.data.copy_(sd["conv_incl.weight"])
    conv_incl.bias.data.copy_(sd["conv_incl.bias"])
    conv_skip.weight.data.copy_(sd["conv_skip.weight"])
    conv_skip.bias.data.copy_(sd["conv_skip.bias"])
    conv_incl.eval(); conv_skip.eval()

    act_incl, act_skip = _compute_kmer_activations(conv_incl, conv_skip, oh_array)
    print(f"  act_incl max={act_incl.max():.4f}  act_skip max={act_skip.max():.4f}")

    # 1. IC logos
    plot_logos(act_incl, oh_array, "incl", out_dir / "kmer_pwm_logos_incl.png")
    plot_logos(act_skip,  oh_array, "skip", out_dir / "kmer_pwm_logos_skip.png")

    # 2. Structural filter heatmaps
    w_struct_incl = sd["conv_struct_incl.weight"].numpy()  # (8, 8, 30)
    w_struct_skip = sd["conv_struct_skip.weight"].numpy()
    plot_struct_heatmaps(w_struct_incl, "incl", out_dir / "struct_heatmaps_incl.png")
    plot_struct_heatmaps(w_struct_skip,  "skip", out_dir / "struct_heatmaps_skip.png")

    # 3. Activation histograms
    plot_activation_hist(act_incl, "incl", out_dir / "kmer_activation_hist_incl.png")
    plot_activation_hist(act_skip,  "skip", out_dir / "kmer_activation_hist_skip.png")

print("\nDone.")
