import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_common
import csv
import os

# Read CSV data
csv_path = os.path.join(os.path.dirname(__file__), "../csv/nvlink_cccl_send_recv.csv")
sizes = []
series = {}

with open(csv_path) as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames[1:]  # skip 'size'
    for col in columns:
        series[col] = []
    for row in reader:
        sizes.append(row["size"])
        for col in columns:
            series[col].append(float(row[col]))

# Convert size labels to numeric MB for log-scale plotting
size_to_mb = {"4MB": 4, "16MB": 16, "64MB": 64, "256MB": 256, "512MB": 512, "1GB": 1024}
sizes_mb = [size_to_mb[s] for s in sizes]

plt.rcParams.update(plot_common.params_line)

fig, ax = plt.subplots(figsize=(6, 5))

sm_list = ["4 SMs", "16 SMs", "64 SMs"]
sm_markers = [plot_common.markers[0], plot_common.markers[1], plot_common.markers[2]]
sm_colors = [plot_common.colors[2], plot_common.colors[3], plot_common.colors[4]]

# LZip-NCCL lines (solid)
for i, sm in enumerate(sm_list):
    key = f"LZip-NCCL ({sm})"
    ax.plot(sizes_mb, series[key],
            marker=sm_markers[i], color=sm_colors[i], linestyle="-")

# NCCL lines (dashed), same markers/colors per SM count
for i, sm in enumerate(sm_list):
    key = f"NCCL ({sm})"
    ax.plot(sizes_mb, series[key],
            marker=sm_markers[i], color=sm_colors[i], linestyle="--")

ax.set_xlabel("Tensor Size")
ax.set_ylabel("Throughput (GB/s)")
ax.set_xscale("log", base=2)

tick_values = sizes_mb
tick_labels = sizes
ax.set_xticks(tick_values)
ax.set_xticklabels(tick_labels, rotation=45, ha="right")
ax.set_xlim(2**1.5, 2**10.5)

# --- Two-group legend ---
# Group 1: method (line style)
method_handles = [
    Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label=plot_common.LABEL_CCCL),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="NCCL"),
]
# Group 2: SM count (marker)
sm_handles = [
    Line2D([0], [0], color=sm_colors[i], marker=sm_markers[i], linestyle="None",
           markersize=8, markerfacecolor="none", markeredgewidth=2, label=sm)
    for i, sm in enumerate(sm_list)
]

leg1 = ax.legend(handles=method_handles, loc="upper left", fontsize="medium",
                 frameon=False, handlelength=2)
ax.add_artist(leg1)
ax.legend(handles=sm_handles, loc="center right", fontsize="medium",
          frameon=False, handlelength=1.5,
          bbox_to_anchor=(1.0, 0.55))

plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "../fig")
plot_common.save_fig(output_dir, "nvlink_cccl_send_recv")
