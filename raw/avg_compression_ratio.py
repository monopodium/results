import re
from collections import defaultdict

input_file = "compression_weight100_ratio.txt"

pattern = re.compile(r'Compressed (\d+) bytes to \d+ bytes, ratio: ([0-9.]+)x')

data = defaultdict(list)

with open(input_file) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            size = int(m.group(1))
            ratio = float(m.group(2))
            data[size].append(ratio)

print(f"{'Size (bytes)':<20} {'Count':<8} {'Avg Ratio':<12}")
print("-" * 40)
for size in sorted(data):
    ratios = data[size]
    avg = sum(ratios) / len(ratios)
    print(f"{size:<20} {len(ratios):<8} {avg:.6f}")
