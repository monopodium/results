import argparse
import matplotlib.pyplot as plt
import plot_common
import numpy as np
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--show-encode-send", action="store_true", default=False,
                    help="Include encode_send series in the plot (omitted by default)")
args = parser.parse_args()

data_file = os.path.join(os.path.dirname(__file__), "../csv/p2p_split_send_vs_2baseline_float16.csv")

data_sizes = []
baseline = []
encode_send = []
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
        data_sizes.append(parse_size(row["Tensor Size"]))
        baseline.append(float(row["Throughput(GB/s) baseline"]))
        encode_send.append(float(row["Throughput(GB/s)encode_send"]))
        split_send.append(float(row["Throughput(GB/s)split_send"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(data_sizes, baseline,    marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Baseline"); idx += 1
if args.show_encode_send:
    plt.plot(data_sizes, encode_send, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Encode Send"); idx += 1
plt.plot(data_sizes, split_send,  marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Split Send")

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [256*1024, 1024**2, 8*1024**2, 128*1024**2, 1024**3]
tick_labels = ["256KB", "1MB", "8MB", "128MB", "1GB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 25, 50, 75])
plt.ylim(0, 80)
plt.legend(loc="upper left", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "p2p_split_send_vs_2baseline_float16")
