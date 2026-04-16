import matplotlib.pyplot as plt
import plot_common
import csv
import os
import numpy as np

data_file = os.path.join(os.path.dirname(__file__), "../csv/dietgpu_walltime_bfloat16.csv")

size_labels = []
comp_throughput = []
decomp_throughput = []

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["Data size"].strip()
        # Remove unnecessary ".0" decimal (e.g. "256.0 KB" -> "256 KB")
        label = label.replace(".0 ", " ")
        size_labels.append(label)
        comp_throughput.append(float(row["comp_throughput_gb_s"]))
        decomp_throughput.append(float(row["decomp_throughput_gb_s"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

x = np.arange(len(size_labels))

idx = 0
plt.plot(x, comp_throughput,   marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Compress"); idx += 1
plt.plot(x, decomp_throughput, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Decompress"); idx += 1

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xticks(x, size_labels, rotation=45, ha="right")
plt.ylim(0, max(max(comp_throughput), max(decomp_throughput)) * 1.15)
plt.legend(loc="upper left", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "dietgpu_walltime_bfloat16")
