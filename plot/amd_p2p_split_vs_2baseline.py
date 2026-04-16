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

# Read data from CSV
data_file = os.path.join(os.path.dirname(__file__), "../csv/amd_p2p_split_vs_2baseline.csv")

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
        data_sizes.append(parse_size(row["Data size"]))
        baseline.append(float(row["baseline"]))
        encode_send.append(float(row["encode_send"]))
        split_send.append(float(row["split_send"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(data_sizes, baseline,    marker=plot_common.markers[idx], color=plot_common.colors[idx], label="UCCL-P2P"); idx += 1
if args.show_encode_send:
    plt.plot(data_sizes, encode_send, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Encode Send"); idx += 1
plt.plot(data_sizes, split_send,  marker=plot_common.markers[idx], color=plot_common.colors[idx], label=plot_common.LABEL_SPLIT_SEND)

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log", base=2)

tick_values = [256*1024, 1024**2, 10*1024**2, 100*1024**2, 400*1024**2]
tick_labels = ["256KB", "1MB", "10MB", "100MB", "400MB"]
plt.xticks(tick_values, tick_labels, rotation=45, ha="right")
plt.yticks([0, 25, 50, 75])
plt.ylim(0, 80)
plt.legend(loc="upper left", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "amd_p2p_split_vs_2baseline")
