import matplotlib.pyplot as plt
import plot_common
import numpy as np
import os

# Data sizes (2 MB to 2 GB)
sizes_mb = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

cccl_throughput_gbps = [
    2.714, 2.945, 3.182, 3.273, 3.387, 3.434,
    3.492, 3.472, 3.45, 3.382, 3.316
]
nccl_matched_throughput_gbps = [
    2.891, 3.346, 3.603, 3.905, 3.968, 3.892,
    3.748, 3.649, 3.676, 3.51, 3.686
]

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

plt.plot(sizes_mb, cccl_throughput_gbps,         marker=plot_common.markers[0], color=plot_common.colors[2], label=plot_common.LABEL_CCCL)
plt.plot(sizes_mb, nccl_matched_throughput_gbps,  marker=plot_common.markers[1], color=plot_common.colors[1], label="NCCL")

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [2, 8, 32, 128, 512, 2048]
tick_labels = ["2MB", "8MB", "32MB", "128MB", "512MB", "2GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 1, 2, 3, 4])
plt.xlim(2**0.7, 2**11.3)
plt.ylim(0, 4.5)
plt.legend(loc="lower right", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "cccl_allreduce")
