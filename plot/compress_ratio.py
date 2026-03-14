import sys
sys.path.insert(0, "/home/ubuntu/efs/shuangma/uep-results/Plot")

import matplotlib.pyplot as plt
import plot_common
import csv
import os

data_file = os.path.join(os.path.dirname(__file__), "../csv/compress_ratio.csv")

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
        bfloat16.append(float(row["compress ratio bfloat16"]))
        float16.append(float(row["compress ratio float16"]))
        float32.append(float(row["compress ratio float32"]))
        float8_e4m3fn.append(float(row["compress ratio float8_e4m3fn"]))
        float8_e5m2.append(float(row["compress ratio float8_e5m2"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

plt.plot(data_sizes, bfloat16,      marker=plot_common.markers[0], color=plot_common.colors[0], label="bfloat16")
plt.plot(data_sizes, float16,       marker=plot_common.markers[1], color=plot_common.colors[1], label="float16")
plt.plot(data_sizes, float32,       marker=plot_common.markers[2], color=plot_common.colors[2], label="float32")
plt.plot(data_sizes, float8_e4m3fn, marker=plot_common.markers[3], color=plot_common.colors[3], label="float8_e4m3fn")
plt.plot(data_sizes, float8_e5m2,   marker=plot_common.markers[4], color=plot_common.colors[4], label="float8_e5m2")

plt.xlabel("Tensor Size")
plt.ylabel("Compression Ratio")
plt.xscale("log", base=2)

tick_values = [16*1024**2, 64*1024**2, 256*1024**2, 1024**3]
tick_labels = ["16MB", "64MB", "256MB", "1GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.ylim(0, 1)
plt.legend(loc="lower right", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "compress_ratio")
