import sys
sys.path.insert(0, "/home/ubuntu/efs/shuangma/uep-results/Plot")

import matplotlib.pyplot as plt
import plot_common
import numpy as np
import csv
import os

# Read data from CSV
data_file = os.path.join(os.path.dirname(__file__), "../csv/nccl_baseline_split_send.csv")

data_sizes = []
nccl = []
baseline = []
split_send = []

def parse_size(s):
    s = s.strip()
    if s.endswith(" B"):
        return float(s[:-2])
    elif s.endswith(" KB"):
        return float(s[:-3]) * 1024
    elif s.endswith(" MB"):
        return float(s[:-3]) * 1024 ** 2
    elif s.endswith(" GB"):
        return float(s[:-3]) * 1024 ** 3
    return float(s)

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_sizes.append(parse_size(row["Data size"]))
        nccl.append(float(row["nccl"]))
        baseline.append(float(row["baseline"]))
        split_send.append(float(row["split_send"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

plt.plot(data_sizes, nccl,       marker=plot_common.markers[0], color=plot_common.colors[0], label="NCCL")
plt.plot(data_sizes, baseline,   marker=plot_common.markers[1], color=plot_common.colors[1], label="Baseline")
plt.plot(data_sizes, split_send, marker=plot_common.markers[2], color=plot_common.colors[2], label="Split Send")

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [256*1024, 1024**2, 8*1024**2, 128*1024**2, 1024**3]
tick_labels = ["256KB", "1MB", "8MB", "128MB", "1GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 25, 50, 75])
plt.legend(loc="upper left", ncol=1)
plt.tight_layout()
plt.ylim(0, 80)

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "nccl_baseline_split_send")
