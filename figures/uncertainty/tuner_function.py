"""Plot the VarianceTuner as a 1D function: z → variance."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
from model import PNASModel

OUT  = Path(__file__).resolve().parent
CKPT = BASE / "checkpoints/flank_150_30_uncertainty/best_model_20260624_193658.pt"

# Observed z ranges (collected via forward hook on variance_bottleneck)
Z_MIN_TRAIN, Z_MAX_TRAIN = -1.415, -0.731
Z_MIN_TEST,  Z_MAX_TEST  = -1.319, -0.719
Z_SWEEP_LO,  Z_SWEEP_HI  = -2.0,   0.0    # wider sweep to show full curve shape

# ── Load model ────────────────────────────────────────────────────────────────
ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PNASModel(input_length=250, use_batchnorm=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── Sweep z through the tuner ─────────────────────────────────────────────────
z_sweep = torch.linspace(Z_SWEEP_LO, Z_SWEEP_HI, 500).unsqueeze(1)  # (500, 1)

with torch.no_grad():
    log_var, var = model.variance_tuner(z_sweep)

z_np   = z_sweep.squeeze().numpy()
var_np = var.squeeze().numpy()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(z_np, var_np, color="#2166ac", linewidth=2)

# Shade train and test observed ranges
ax.axvspan(Z_MIN_TRAIN, Z_MAX_TRAIN, alpha=0.12, color="#4dac26",
           label=f"Train range  [{Z_MIN_TRAIN:.2f}, {Z_MAX_TRAIN:.2f}]")
ax.axvspan(Z_MIN_TEST,  Z_MAX_TEST,  alpha=0.20, color="#d6604d",
           label=f"Test range   [{Z_MIN_TEST:.2f},  {Z_MAX_TEST:.2f}]")

ax.set_xlabel("Bottleneck output", fontsize=9)
ax.set_ylabel("Predicted variance", fontsize=9)
ax.set_title("Variance tuner", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()

out_path = OUT / "tuner_function.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
print(f"Variance at train z_min ({Z_MIN_TRAIN}): {np.interp(Z_MIN_TRAIN, z_np, var_np):.4f}")
print(f"Variance at train z_max ({Z_MAX_TRAIN}): {np.interp(Z_MAX_TRAIN, z_np, var_np):.4f}")
print(f"Variance at test  z_min ({Z_MIN_TEST}):  {np.interp(Z_MIN_TEST,  z_np, var_np):.4f}")
print(f"Variance at test  z_max ({Z_MAX_TEST}):  {np.interp(Z_MAX_TEST,  z_np, var_np):.4f}")
