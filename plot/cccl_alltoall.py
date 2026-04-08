import matplotlib.pyplot as plt
import plot_common
import numpy as np
import os

# Data sizes (128 KB to 2 GB)
sizes_mb = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

cccl_throughput_gbps = [
    1.809, 2.261, 3.514, 4.109, 4.777, 4.799, 4.8,
    4.996, 5.09, 5.174, 5.244, 5.388, 5.389, 5.278, 5.189
]
nccl_matched_throughput_gbps = [
    1.863, 2.198, 3.483, 4.244, 4.634, 4.593, 4.622,
    4.548, 4.534, 4.495, 4.52, 4.501, 4.448, 4.482, 4.439
]

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

plt.plot(sizes_mb, cccl_throughput_gbps,         marker=plot_common.markers[0], color=plot_common.colors[2], label=plot_common.LABEL_CCCL)
plt.plot(sizes_mb, nccl_matched_throughput_gbps,  marker=plot_common.markers[1], color=plot_common.colors[1], label="NCCL")

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [0.125, 1, 8, 128, 1024]
tick_labels = ["128KB", "1MB", "8MB", "128MB", "1GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 2, 4, 6])
plt.xlim(2**(-3.8), 2**11)
plt.ylim(0, 7)
plt.legend(loc="lower right", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "cccl_alltoall")
