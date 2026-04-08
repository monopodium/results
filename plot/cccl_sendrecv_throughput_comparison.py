import matplotlib.pyplot as plt
import plot_common
import numpy as np
import os

# Data sizes in MB
sizes_mb = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Throughput data (GB/s)
cccl_throughput_gbps = [
    1.793, 3.068, 3.316, 4.221, 4.899, 5.109, 5.333,
    5.657, 5.815, 5.854, 6.015, 6.051, 6.111, 6.057, 6.121
]
nccl_matched_throughput_gbps = [
    1.193, 1.847, 3.139, 3.908, 4.622, 4.633, 4.511,
    4.535, 4.540, 4.553, 4.493, 4.519, 4.499, 4.465, 4.459
]
nccl_default_throughput_gbps = [
    1.125, 2.344, 3.293, 4.128, 4.538, 4.468, 4.582,
    4.576, 4.548, 4.525, 4.478, 4.543, 4.461, 4.502, 4.489
]

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

n = 14  # plot up to 1024 MB, exclude last point (2048 MB)
plt.plot(sizes_mb[:n], cccl_throughput_gbps[:n],        marker=plot_common.markers[0], color=plot_common.colors[2], label=plot_common.LABEL_CCCL)
plt.plot(sizes_mb[:n], nccl_matched_throughput_gbps[:n], marker=plot_common.markers[1], color=plot_common.colors[1], label="NCCL")

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [0.125, 1, 8,128, 1024]
tick_labels = ["128KB", "1MB", "8MB", "128MB", "1GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 2, 4, 6])
plt.xlim(2**(-3.8), 2**11)   # 128KB to ~2GB, padding on both sides
plt.ylim(0, 7)
plt.legend(loc="lower right", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "cccl_sendrecv_throughput_comparison")
