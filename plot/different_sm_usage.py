import argparse
import matplotlib.pyplot as plt
import plot_common
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--show-encode-send", action="store_true", default=False,
                    help="Include encode_send series in the plot (omitted by default)")
args = parser.parse_args()

data_file = os.path.join(os.path.dirname(__file__), "../csv/different_sm_usage.csv")

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
baseline = []
sm_100 = []
sm_50 = []
sm_75 = []

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_sizes.append(parse_size(row["Data size"]))
        baseline.append(float(row["baseline"]))
        sm_100.append(float(row["100% SM"]))
        sm_50.append(float(row["50% SM"]))
        sm_75.append(float(row["75% SM "]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(data_sizes, baseline, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="UCCL-P2P"); idx += 1
plt.plot(data_sizes, sm_100,   marker=plot_common.markers[idx], color=plot_common.colors[idx], label="100% SM"); idx += 1
plt.plot(data_sizes, sm_75,    marker=plot_common.markers[idx], color=plot_common.colors[idx], label="75% SM"); idx += 1
plt.plot(data_sizes, sm_50,    marker=plot_common.markers[idx], color=plot_common.colors[idx], label="50% SM"); idx += 1

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
plot_common.save_fig(output_dir, "different_sm_usage")
