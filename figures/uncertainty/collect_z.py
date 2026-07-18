"""Collect variance bottleneck z values for train and test sets. Requires torch."""

from pathlib import Path
import numpy as np
import torch
import sys

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
from model import PNASModel

CKPT  = BASE / "checkpoints/flank_150_30_uncertainty/best_model_20260624_193658.pt"
OUT   = Path(__file__).resolve().parent
BATCH = 512

ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PNASModel(input_length=250, use_batchnorm=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def collect_z(npz_path: Path) -> np.ndarray:
    data    = np.load(npz_path)
    seq_oh  = torch.tensor(data["seq_oh"],    dtype=torch.float32)
    str_oh  = torch.tensor(data["struct_oh"], dtype=torch.float32)
    wobbles = torch.tensor(data["wobbles"],   dtype=torch.float32)
    z_vals  = []
    def hook(module, inp, out):
        z_vals.append(out.detach().cpu())
    handle = model.variance_bottleneck.register_forward_hook(hook)
    with torch.no_grad():
        for i in range(0, len(seq_oh), BATCH):
            model(seq_oh[i:i+BATCH], str_oh[i:i+BATCH], wobbles[i:i+BATCH],
                  return_uncertainty=True)
    handle.remove()
    return torch.cat(z_vals).squeeze().numpy()

print("Running train set...")
z_train = collect_z(BASE / "data/train_flank_150_30.npz")
print("Running test set...")
z_test  = collect_z(BASE / "data/test_flank_150_30.npz")
print(f"Train z: [{z_train.min():.3f}, {z_train.max():.3f}]  n={len(z_train)}")
print(f"Test  z: [{z_test.min():.3f},  {z_test.max():.3f}]  n={len(z_test)}")

np.save(OUT / "z_train.npy", z_train)
np.save(OUT / "z_test.npy",  z_test)
print("Saved z_train.npy and z_test.npy")
