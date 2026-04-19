import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/test_data.csv")
df = df[["PSI", "predicted_PSI", "predicted_mfe"]].copy()
#print(df.info())

eps = 1e-6
# KL divergence for each exon
P = np.clip(df["PSI"].to_numpy(), eps, 1 - eps)
Q = np.clip(df["predicted_PSI"].to_numpy(), eps, 1 - eps)
df["kl"] = P * np.log(P / Q) + (1 - P) * np.log((1 - P) / (1 - Q))

# bins by MFE
df['bin'] = pd.qcut(df["predicted_mfe"], q=20, labels=False, duplicates="drop")
df["bin"] = df["bin"] + 1 # shift index
bin_loss = df.groupby("bin")["kl"].mean()
#print(bin_loss)

# plot
plt.close("all")
fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    bin_loss.index,
    bin_loss.values,
    marker="o",
    linewidth=2.5,
    markersize=7
)

ax.set_title("Model loss vs MFE bin", fontsize=18, pad=12)
ax.set_xlabel("MFE bin", fontsize=13)
ax.set_ylabel("Mean KL divergence", fontsize=13)

ax.set_xticks(range(1, 21))
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()