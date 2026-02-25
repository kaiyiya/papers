"""
绘制训练收敛曲线（论文用）
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = "versions/v7.2_xy_balanced/logs/training_history_v7_2_20260205_061717.json"
OUT = "versions/v7.2_xy_balanced/training_curve.pdf"

with open(LOG) as f:
    h = json.load(f)

train_loss = np.array(h["train_loss"])
val_loss   = np.array(h.get("val_loss", h.get("test_loss", [])))
epochs = np.arange(1, len(train_loss) + 1)

val_loss        = np.array(h["val_loss"])
train_path_loss = np.array(h["train_path_loss"])
val_path_loss   = np.array(h["val_path_loss"])
train_cov_loss  = np.array(h["train_coverage_loss"])
val_cov_loss    = np.array(h["val_coverage_loss"])
tf_ratio        = np.array(h["tf_ratio"])

# val 可能每N个epoch记录一次
val_epochs = np.linspace(1, len(train_loss), len(val_loss))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- (a) Total loss ---
ax = axes[0]
ax.plot(epochs, -train_loss, label="Train", color="#2196F3", linewidth=1.8)
ax.plot(val_epochs, -val_loss,   label="Val",   color="#FF5722", linewidth=1.8, linestyle="--")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("SoftDTW Loss", fontsize=12)
ax.set_title("(a) Total Loss", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# --- (b) Path loss vs Coverage loss ---
ax = axes[1]
ax.plot(epochs, -train_path_loss, label="Train Path", color="#2196F3", linewidth=1.8)
ax.plot(val_epochs, -val_path_loss,   label="Val Path",   color="#FF5722", linewidth=1.8, linestyle="--")
ax.plot(epochs, -train_cov_loss,  label="Train Cov",  color="#4CAF50", linewidth=1.4, linestyle=":")
ax.plot(val_epochs, -val_cov_loss,    label="Val Cov",    color="#FF9800", linewidth=1.4, linestyle="-.")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("(b) Path & Coverage Loss", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- (c) Teacher forcing ratio ---
ax = axes[2]
ax.plot(epochs, tf_ratio, color="#9C27B0", linewidth=1.8)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Teacher Forcing Ratio", fontsize=12)
ax.set_title("(c) Teacher Forcing Schedule", fontsize=13)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
