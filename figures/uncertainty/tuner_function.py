"""Plot the VarianceTuner as a 1D function: z → variance, with z distributions."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))

OUT  = Path(__file__).resolve().parent

Z_SWEEP_LO, Z_SWEEP_HI = -2.0, 0.0

# ── Sweep z through the tuner (numpy, no torch needed) ───────────────────────
# Tuner is: softplus(linear(z)) → exp → variance. We load saved z arrays and
# use the precomputed curve from a simple numpy reconstruction.
# Since we don't have torch here, load the saved sweep arrays instead.
# (Run collect_z.py first with anaconda python to generate z_train.npy / z_test.npy)
try:
    import torch
    from model import PNASModel
    CKPT = BASE / "checkpoints/flank_150_30_uncertainty/best_model_20260624_193658.pt"
    ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = PNASModel(input_length=250, use_batchnorm=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    z_sweep = torch.linspace(Z_SWEEP_LO, Z_SWEEP_HI, 500).unsqueeze(1)
    with torch.no_grad():
        _, var = model.variance_tuner(z_sweep)
    z_np   = z_sweep.squeeze().numpy()
    var_np = var.squeeze().numpy()
    np.save(OUT / "z_sweep.npy",   z_np)
    np.save(OUT / "var_sweep.npy", var_np)
except ModuleNotFoundError:
    z_np   = np.load(OUT / "z_sweep.npy")
    var_np = np.load(OUT / "var_sweep.npy")

# ── Load precomputed z distributions (from collect_z.py) ──────────────────────
z_train = np.load(OUT / "z_train.npy")
z_test  = np.load(OUT / "z_test.npy")
print(f"Train z: [{z_train.min():.3f}, {z_train.max():.3f}]  n={len(z_train)}")
print(f"Test  z: [{z_test.min():.3f},  {z_test.max():.3f}]  n={len(z_test)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax2 = ax.twinx()   # second y-axis for histograms

# Histograms on ax2 (behind the curve)
bins = np.linspace(Z_SWEEP_LO, Z_SWEEP_HI, 80)
ax2.hist(z_train, bins=bins, color="#4dac26", alpha=0.25, label="Train dist")
ax2.hist(z_test,  bins=bins, color="#d6604d", alpha=0.25, label="Test dist")
ax2.set_ylabel("Count", fontsize=9, color="#888888")
ax2.tick_params(axis="y", labelcolor="#888888")
ax2.set_zorder(1)

# Variance curve on ax (on top)
ax.set_zorder(2)
ax.patch.set_visible(False)
ax.plot(z_np, var_np, color="#2166ac", linewidth=2, label="Variance tuner")

# Shaded ranges
z_min_train, z_max_train = z_train.min(), z_train.max()
z_min_test,  z_max_test  = z_test.min(),  z_test.max()
ax.axvspan(z_min_train, z_max_train, alpha=0.10, color="#4dac26",
           label=f"Train range  [{z_min_train:.2f}, {z_max_train:.2f}]")
ax.axvspan(z_min_test,  z_max_test,  alpha=0.15, color="#d6604d",
           label=f"Test range   [{z_min_test:.2f},  {z_max_test:.2f}]")

ax.set_xlabel("Bottleneck output", fontsize=9)
ax.set_ylabel("Predicted variance", fontsize=9)
ax.set_title("Variance tuner", fontsize=10)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

fig.tight_layout()
out_path = OUT / "tuner_function.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ── Zoomed plot: -1.5 to -0.5 ─────────────────────────────────────────────────
ZOOM_LO, ZOOM_HI = -1.5, -0.5

fig2, ax_z = plt.subplots(figsize=(7, 4))
ax2_z = ax_z.twinx()

bins_zoom = np.linspace(ZOOM_LO, ZOOM_HI, 120)
ax2_z.hist(z_train, bins=bins_zoom, color="#4dac26", alpha=0.25, label="Train dist")
ax2_z.hist(z_test,  bins=bins_zoom, color="#d6604d", alpha=0.25, label="Test dist")
ax2_z.set_ylabel("Count", fontsize=9, color="#888888")
ax2_z.tick_params(axis="y", labelcolor="#888888")
ax2_z.set_zorder(1)

ax_z.set_zorder(2)
ax_z.patch.set_visible(False)

mask = (z_np >= ZOOM_LO) & (z_np <= ZOOM_HI)
ax_z.plot(z_np[mask], var_np[mask], color="#2166ac", linewidth=2, label="Variance tuner")
ax_z.axvspan(z_min_train, z_max_train, alpha=0.10, color="#4dac26",
             label=f"Train range  [{z_min_train:.2f}, {z_max_train:.2f}]")
ax_z.axvspan(z_min_test,  z_max_test,  alpha=0.15, color="#d6604d",
             label=f"Test range   [{z_min_test:.2f},  {z_max_test:.2f}]")

ax_z.set_xlim(ZOOM_LO, ZOOM_HI)
ax2_z.set_xlim(ZOOM_LO, ZOOM_HI)
ax_z.set_xlabel("Bottleneck output", fontsize=9)
ax_z.set_ylabel("Predicted variance", fontsize=9)
ax_z.set_title("Variance tuner (zoomed)", fontsize=10)

lines1, labels1 = ax_z.get_legend_handles_labels()
lines2, labels2 = ax2_z.get_legend_handles_labels()
ax_z.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

fig2.tight_layout()
out_zoom = OUT / "tuner_function_zoom.png"
fig2.savefig(out_zoom, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_zoom}")
