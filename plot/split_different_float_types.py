import matplotlib.pyplot as plt
import plot_common
import csv
import os

data_file = os.path.join(os.path.dirname(__file__), "../csv/split_different_float_types.csv")

data_sizes = []
bfloat16 = []
float16 = []
float32 = []
float8_e4m3fn = []
float8_e5m2 = []

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
        bfloat16.append(float(row["Throughput(GB/s) bfloat16"]))
        float16.append(float(row["Throughput(GB/s) float16"]))
        float32.append(float(row["Throughput(GB/s) float32"]))
        float8_e4m3fn.append(float(row["Throughput(GB/s) float8_e4m3fn"]))
        float8_e5m2.append(float(row["Throughput(GB/s) float8_e5m2"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

plt.plot(data_sizes, bfloat16,      marker=plot_common.markers[0], color=plot_common.colors[0], label="bfloat16")
plt.plot(data_sizes, float16,       marker=plot_common.markers[1], color=plot_common.colors[1], label="float16")
plt.plot(data_sizes, float32,       marker=plot_common.markers[2], color=plot_common.colors[2], label="float32")
plt.plot(data_sizes, float8_e4m3fn, marker=plot_common.markers[3], color=plot_common.colors[3], label="float8_e4m3fn")
plt.plot(data_sizes, float8_e5m2,   marker=plot_common.markers[4], color=plot_common.colors[4], label="float8_e5m2")

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [256*1024, 1024**2, 8*1024**2, 128*1024**2, 1024**3]
tick_labels = ["256KB", "1MB", "8MB", "128MB", "1GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 25, 50, 75])
plt.ylim(0, 80)
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
leg1 = ax.legend(handles[:3], labels[:3], loc="upper left", ncol=1)
ax.add_artist(leg1)
ax.legend(handles[3:], labels[3:], loc="lower right", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "split_different_float_types")
