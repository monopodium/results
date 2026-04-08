import matplotlib.pyplot as plt
import plot_common
import numpy as np
import os

# Data (32 KB to 1 GB)
sizes_mb = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# ZeroGPU compression ratio data
zerogpu_compression_ratio = [
    0.7349, 0.7346, 0.7347, 0.7346, 0.7348, 0.7352, 0.7359, 0.7368,
    0.7380, 0.7305, 0.7265, 0.7245, 0.7235, 0.7230, 0.7228, 0.7227
]

# DietGPU compression ratio data
dietgpu_compression_ratio = [
    0.6899, 0.6814, 0.6772, 0.6752, 0.6743, 0.6741, 0.6740, 0.6741,
    0.6740, 0.6741, 0.6742, 0.6744, 0.6745, 0.6745, 0.6745, 0.6744
]

# Filter to selected sizes: 256 KB, 1 MB, 8 MB, 16 MB, 32 MB, 64 MB
selected_mb = [0.25, 1, 8, 16, 32, 64]
indices = [sizes_mb.index(s) for s in selected_mb]
sel_zerogpu = [zerogpu_compression_ratio[i] for i in indices]
sel_dietgpu = [dietgpu_compression_ratio[i] for i in indices]
plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

x = np.arange(len(selected_mb))
tick_labels = ["256KB", "1MB", "8MB", "16MB", "32MB", "64MB"]

plt.plot(x, sel_zerogpu, marker=plot_common.markers[0], color=plot_common.colors[0], label="Global Frequency Table")
plt.plot(x, sel_dietgpu, marker=plot_common.markers[1], color=plot_common.colors[1], label="Localized Frequency Tables")

plt.xlabel("Tensor Size")
plt.ylabel("Compression Ratio")

plt.xticks(x, tick_labels, rotation=45, ha="right")

plt.ylim(0, 1.0)

plt.legend(loc="lower center", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "compression_ratio_by_size_h200")
