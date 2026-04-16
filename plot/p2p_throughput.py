import matplotlib.pyplot as plt
import plot_common
import csv
import os

data_file = os.path.join(os.path.dirname(__file__), "../csv/p2p_throughput.csv")

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

data_labels = []
uccl_throughput = []
nccl_throughput = []

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_labels.append(row["Data size"].strip())
        uccl_throughput.append(float(row["Throughput (GB/s) UCCL-P2P"]))
        nccl_throughput.append(float(row["Throughput (GB/s) NCCL"]))

x = list(range(len(data_labels)))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(x, uccl_throughput, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="UCCL-P2P"); idx += 1
plt.plot(x, nccl_throughput, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="NCCL"); idx += 1

plt.xlabel("Data Size")
plt.ylabel("Throughput (GB/s)")

tick_labels = [l.replace(".0 ", " ") for l in data_labels]
plt.xticks(x, tick_labels, rotation=45, ha="right")
plt.ylim(0, 50)
plt.legend(loc="upper left", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "p2p_throughput")
