import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, "/home/ubuntu/efs/shuangma/uep-results/Plot")
import plot_common

input_tokens = [7680, 10240, 15360, 25600, 51200, 76800, 102400]
nccl_latency = [221, 306, 427, 689, 1441, 1970, 2637]
cccl_latency = [209, 258, 358, 535, 1108, 1623, 2133]

plt.rcParams.update(plot_common.params_line)

x = np.arange(len(input_tokens))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(x - width / 2, nccl_latency, width, label='NCCL', color=plot_common.colors[0], hatch=plot_common.hatches[0])
ax.bar(x + width / 2, cccl_latency, width, label='CCCL', color=plot_common.colors[1], hatch=plot_common.hatches[1])

ax.set_xlabel('Input Token')
ax.set_ylabel('Latency (ms)')
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in input_tokens], rotation=30, ha='right')
ax.legend(loc='upper left')
plt.tight_layout()

output_dir = "/home/ubuntu/efs/shuangma/results/fig"
plot_common.save_fig(output_dir, "cccl_p2p_pd_disaggregation")
