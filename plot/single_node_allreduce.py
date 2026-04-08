import matplotlib.pyplot as plt
import plot_common
import csv
import os

data_file = os.path.join(os.path.dirname(__file__), "../csv/single_node_allreduce.csv")

data_sizes = []
nccl = []
lzip = []
two_shot = []

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
        data_sizes.append(parse_size(row["Tensor Size"]))
        nccl.append(float(row["NCCL (Ring) GB/s"]))
        lzip.append(float(row["LZip Compression GB/s"]))
        two_shot.append(float(row["Two Shot GB/s"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(data_sizes, nccl,     marker=plot_common.markers[idx], color=plot_common.colors[idx], label="NCCL (Ring)"); idx += 1
plt.plot(data_sizes, lzip,     marker=plot_common.markers[idx], color=plot_common.colors[idx], label="LZip Two-Shot"); idx += 1
plt.plot(data_sizes, two_shot, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Two-Shot")

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [8*1024**2, 32*1024**2, 128*1024**2, 512*1024**2, 1024**3]
tick_labels = ["8MB", "32MB", "128MB", "512MB", "1GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 10, 20, 30, 40])
plt.legend(loc="upper left", ncol=1)
plt.tight_layout()
plt.ylim(0, 40)

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "single_node_allreduce")
