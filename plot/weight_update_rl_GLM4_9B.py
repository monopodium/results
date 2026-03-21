import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import plot_common

parser = argparse.ArgumentParser()
parser.add_argument("--show-encode-send", action="store_true", default=False,
                    help="Include encode_send series in the plot (omitted by default)")
args = parser.parse_args()

plt.rcParams.update(plot_common.params_line)

csv_path = "/home/ubuntu/efs/shuangma/results/csv/weight_update_rl_GLM4_9B.csv"
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
df["Weights"] = df["Weights"].str.strip()

# Build x-axis labels: "weight_name(size MB)", strip trailing .weight
labels = [f"{w.removesuffix('.weight')}\n({int(s)} MB)" for w, s in zip(df["Weights"], df["Size (MB)"])]

# Find throughput columns by stripping spaces
col_map = {c.replace(" ", "").lower(): c for c in df.columns}
baseline_col   = col_map["throughput(gb/s)baseline"]
encode_col     = col_map["throughput(gb/s)encode_send"]
split_col      = col_map["throughput(gb/s)split_send"]
ratio_col      = col_map["compression_ratio"]

baseline   = df[baseline_col].values
encode     = df[encode_col].values
split      = df[split_col].values
ratio      = df[ratio_col].values

x = np.arange(len(labels))

if args.show_encode_send:
    width = 0.25
    offsets = [-width, 0, width]
else:
    width = 0.3
    offsets = [-width/2, width/2]

fig, ax1 = plt.subplots(figsize=(10, 5))

# --- Bars (left axis: throughput) ---
bars_handles = []
bars_labels_list = []

b1 = ax1.bar(x + offsets[0], baseline, width,
             label="Baseline",
             color=plot_common.colors[0], hatch=plot_common.hatches[0])
bars_handles.append(b1)
bars_labels_list.append("Baseline")

bi = 1
if args.show_encode_send:
    b2 = ax1.bar(x + offsets[bi], encode, width,
                 label="Encode Send",
                 color=plot_common.colors[1], hatch=plot_common.hatches[1])
    bars_handles.append(b2)
    bars_labels_list.append("Encode Send")
    bi += 1

b3 = ax1.bar(x + offsets[bi], split, width,
             label=plot_common.LABEL_SPLIT_SEND,
             color=plot_common.colors[2], hatch=plot_common.hatches[2])
bars_handles.append(b3)
bars_labels_list.append(plot_common.LABEL_SPLIT_SEND)

ax1.set_xlabel("Weights Tensor")
ax1.set_ylabel("Throughput (GB/s)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=13)
ax1.set_ylim(0, max(split) * 1.35)

# --- Line (right axis: compression ratio) ---
ax2 = ax1.twinx()
line, = ax2.plot(x, ratio, color="lightgray",
                 marker=plot_common.markers[0],
                 linestyle="-", linewidth=2, markersize=8,
                 label="compression_ratio %")
ax2.set_ylabel("Compression Ratio")
ax2.set_ylim(0, 1)

# --- Legend: merge both axes ---
ax1.legend(bars_handles + [line],
           bars_labels_list + ["compression_ratio"],
           loc="upper left", ncol=2)

plt.tight_layout()

output_dir = "/home/ubuntu/efs/shuangma/results/fig"
os.makedirs(output_dir, exist_ok=True)
plot_common.save_fig(output_dir, "weight_update_rl_GLM4_9B")
