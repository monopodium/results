import re

input_file = "compression_qwen3_5_35B_A3B.txt"

compress_pattern = re.compile(r'Compressed (\d+) bytes to \d+ bytes, ratio: ([0-9.]+)x')
client_pattern = re.compile(r'\[Client\]\s+(model\.\S+)\s+([\d.]+)\s+MB')

# Accumulate compression ratios; assign to the weight name when [Client] line is hit
buffer = []  # list of (size_bytes, ratio)
results = {}  # weight_name -> list of (size_bytes, ratio)

with open(input_file) as f:
    for line in f:
        m = compress_pattern.search(line)
        if m:
            size = int(m.group(1))
            ratio = float(m.group(2))
            buffer.append((size, ratio))
            continue

        m = client_pattern.search(line)
        if m:
            weight_name = m.group(1)
            if weight_name not in results:
                results[weight_name] = []
            results[weight_name].extend(buffer)
            buffer = []

print(f"{'Weight':<45} {'Size (MB)':<12} {'Count':<8} {'Avg Ratio':<12}")
print("-" * 77)
for weight, entries in results.items():
    ratios = [r for _, r in entries]
    size_bytes = entries[0][0] if entries else 0
    size_mb = size_bytes / (1024 * 1024)
    avg = sum(ratios) / len(ratios)
    print(f"{weight:<45} {size_mb:<12.1f} {len(ratios):<8} {avg:.6f}")
