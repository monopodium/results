import matplotlib.pyplot as plt
import plot_common
import csv
import os

data_file = os.path.join(os.path.dirname(__file__), "../csv/dietgpu_walltime_float8_e4m3fn.csv")

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

data_sizes = []
comp_throughput = []
decomp_throughput = []

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_sizes.append(parse_size(row["Data size"]))
        comp_throughput.append(float(row["comp_throughput_gb_s"]))
        decomp_throughput.append(float(row["decomp_throughput_gb_s"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(data_sizes, comp_throughput,   marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Compress"); idx += 1
plt.plot(data_sizes, decomp_throughput, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Decompress"); idx += 1

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [512*1024, 1024**2, 4*1024**2, 16*1024**2, 64*1024**2]
tick_labels = ["512KB", "1MB", "4MB", "16MB", "64MB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.ylim(0, 550)
plt.legend(loc="upper left", ncol=1)
plt.title("DietGPU Float Codec (float8_e4m3fn)")
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "dietgpu_walltime_float8_e4m3fn")
