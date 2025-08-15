import matplotlib.pyplot as plt
import numpy as np

# Epochs chi tiết: khoảng cách 1, tổng 1000 epoch
epochs = list(range(0, 1000, 1))

# Dữ liệu giả lập với nhiễu cao hơn
np.random.seed(42)

precision = np.clip(np.linspace(40, 65, len(epochs)) + np.random.normal(0, 4, len(epochs)), 30, 75)
recall = np.clip(np.linspace(45, 68, len(epochs)) + np.random.normal(0, 4, len(epochs)), 35, 78)
map_50 = np.clip(np.linspace(50, 70, len(epochs)) + np.random.normal(0, 4, len(epochs)), 40, 80)
map_5095 = np.clip(np.linspace(25, 45, len(epochs)) + np.random.normal(0, 3.5, len(epochs)), 15, 55)

# Losses giả lập (giảm dần nhưng nhiễu nhiều hơn)
train_box = np.clip(np.linspace(0.04, 0.02, len(epochs)) + np.random.normal(0, 0.006, len(epochs)), 0.015, 0.05)
train_obj = np.clip(np.linspace(0.03, 0.015, len(epochs)) + np.random.normal(0, 0.006, len(epochs)), 0.01, 0.04)
train_cls = np.clip(np.linspace(0.02, 0.005, len(epochs)) + np.random.normal(0, 0.003, len(epochs)), 0.002, 0.03)

val_box = train_box - np.random.normal(0.003, 0.0025, len(epochs))
val_obj = train_obj - np.random.normal(0.003, 0.0025, len(epochs))
val_cls = train_cls - np.random.normal(0.0015, 0.0015, len(epochs))

# Hàm vẽ
def plot(ax, x, y, title):
    ax.plot(x, y, label='results', color='blue', linewidth=0.9, alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Epochs', fontsize=8)
    ax.set_ylabel('Value', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)

# Vẽ biểu đồ
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Fake Training Metrics (~60% Precision, More Noise)', fontsize=16)

# Hàng trên
plot(axs[0, 0], epochs, train_box, 'train/box_loss')
plot(axs[0, 1], epochs, train_obj, 'train/obj_loss')
plot(axs[0, 2], epochs, train_cls, 'train/cls_loss')
plot(axs[0, 3], epochs, precision, 'metrics/precision')
plot(axs[0, 4], epochs, recall, 'metrics/recall')

# Hàng dưới
plot(axs[1, 0], epochs, val_box, 'val/box_loss')
plot(axs[1, 1], epochs, val_obj, 'val/obj_loss')
plot(axs[1, 2], epochs, val_cls, 'val/cls_loss')
plot(axs[1, 3], epochs, map_50, 'metrics/mAP_0.5')
plot(axs[1, 4], epochs, map_5095, 'metrics/mAP_0.5:0.95')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fake_training_metrics_blue_noisy.png', dpi=200)
plt.show()
