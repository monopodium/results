import matplotlib.pyplot as plt
import plot_common
import csv
import os

data_file = os.path.join(os.path.dirname(__file__), "../csv/allreduce_throughput.csv")

data_labels = []
nccl_throughput = []
twoshot_throughput = []
lzip_throughput = []

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_labels.append(row["Data size"].strip())
        nccl_throughput.append(float(row["Throughput (GB/s) NCCL"]))
        twoshot_throughput.append(float(row["Throughput (GB/s) Two-Shot"]))
        lzip_throughput.append(float(row["Throughput (GB/s) LZip Two-Shot"]))

x = list(range(len(data_labels)))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(x, lzip_throughput, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="LZip Two-Shot"); idx += 1
plt.plot(x, twoshot_throughput, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Two-Shot"); idx += 1
plt.plot(x, nccl_throughput, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="NCCL"); idx += 1

plt.xlabel("Data Size")
plt.ylabel("Throughput (GB/s)")

tick_labels = [l.replace(".0 ", " ") for l in data_labels]
plt.xticks(x, tick_labels, rotation=45, ha="right")
plt.ylim(0, 60)
plt.legend(loc="upper left", ncol=1)
plt.title("AllReduce Throughput")
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "allreduce_throughput")
