import matplotlib.pyplot as plt
import plot_common
import csv
import os

data_file = os.path.join(os.path.dirname(__file__), "../csv/compare_with_native_pipline.csv")

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
split_send = []
naive_pipeline = []
encode_send = []

with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_sizes.append(parse_size(row["Data size"]))
        baseline.append(float(row["baseline"]))
        split_send.append(float(row["split_send"]))
        naive_pipeline.append(float(row["naive pipeline"]))
        encode_send.append(float(row["encode_send"]))

plt.rcParams.update(plot_common.params_line)

plt.figure(figsize=(6, 5))

idx = 0
plt.plot(data_sizes, baseline,       marker=plot_common.markers[idx], color=plot_common.colors[idx], label="UCCL-P2P"); idx += 1
plt.plot(data_sizes, encode_send,    marker=plot_common.markers[idx], color=plot_common.colors[idx], label="encode-send"); idx += 1
plt.plot(data_sizes, split_send,     marker=plot_common.markers[idx], color=plot_common.colors[idx], label=plot_common.LABEL_SPLIT_SEND); idx += 1
plt.plot(data_sizes, naive_pipeline, marker=plot_common.markers[idx], color=plot_common.colors[idx], label="naive pipeline"); idx += 1

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
plot_common.save_fig(output_dir, "compare_with_native_pipeline")
