import argparse
import matplotlib.pyplot as plt
import plot_common
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--show-encode-send", action="store_true", default=False,
                    help="Include encode_send series in the plot (omitted by default)")
args = parser.parse_args()

data_file = os.path.join(os.path.dirname(__file__), "../csv/different_memory_usage.csv")

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
baseline = []
lzip_p2p = []
chunked_lzip_p2p = []

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_labels.append(row["Data size"].strip())
        baseline.append(float(row["baseline"]))
        lzip_p2p.append(float(row["LZip-P2P"]))
        chunked_lzip_p2p.append(float(row["Chunked LZip-P2P"]))

x = list(range(len(data_labels)))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
l_baseline, = plt.plot(x, baseline,          marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Baseline"); idx += 1
l_lzip,     = plt.plot(x, lzip_p2p,          marker=plot_common.markers[idx], color=plot_common.colors[idx], label="LZip-P2P"); idx += 1
l_chunked,  = plt.plot(x, chunked_lzip_p2p,  marker=plot_common.markers[idx], color=plot_common.colors[idx], label="Chunked LZip-P2P"); idx += 1

plt.xlabel("Tensor Size")
plt.ylabel("Throughput (GB/s)")

tick_labels = [l.replace(".0 ", " ") for l in data_labels]
plt.xticks(x, tick_labels, rotation=45, ha="right")
plt.yticks([0, 25, 50, 75])
plt.ylim(0, 80)
plt.legend(handles=[l_baseline, l_chunked, l_lzip], loc="lower right", ncol=1)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "different_memory_usage")
