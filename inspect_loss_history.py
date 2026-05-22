import torch
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load checkpoint
# =========================

#ckpt_path = "model_weights.pt"
ckpt_path = "checkpoints/vienna60/best_model_20260515_150124.pt"

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False
)

# =========================
# Basic info
# =========================
print("Checkpoint keys:")
print(ckpt.keys())

print("\nBest validation loss:")
print(ckpt["best_val_loss"])

print("\nCheckpoint saved at epoch:")
print(ckpt["epoch"])

# =========================
# Training history
# =========================
history = pd.DataFrame(ckpt["history"])

print("\nTraining history:")
print(history)

print("\nMinimum validation loss:")
print(history["val_loss"].min())

print("\nFinal validation loss:")
print(history["val_loss"].iloc[-1])

# =========================
# Plot training curves
# =========================
plt.figure(figsize=(7,5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Train loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("KL loss")
plt.title("Training curves")

plt.legend()

plt.tight_layout()
plt.show()