#!/bin/bash
# Run the benchmark across 2 nodes.
#
# Option 1: torchrun (recommended)
#   On node 0:
#     torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
#       --master_addr=<MASTER_IP> --master_port=29500 benchmark_both.py
#   On node 1:
#     torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
#       --master_addr=<MASTER_IP> --master_port=29500 benchmark_both.py
#
# Option 2: Manual env vars
#   On node 0:
#     MASTER_ADDR=<MASTER_IP> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python benchmark_both.py
#   On node 1:
#     MASTER_ADDR=<MASTER_IP> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python benchmark_both.py
#
# Option 3: Single-node test with 2 GPUs (for quick local testing)
#   torchrun --nproc_per_node=2 --nnodes=1 benchmark_both.py

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <master_addr> [node_rank (0 or 1)]"
    echo ""
    echo "Single-node 2-GPU test:"
    echo "  $0 local"
    exit 1
fi

MASTER_ADDR=$1
NODE_RANK=${2:-0}
MASTER_PORT=${MASTER_PORT:-29500}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$MASTER_ADDR" = "local" ]; then
    echo "=== Single-node 2-GPU test ==="
    torchrun --nproc_per_node=2 --nnodes=1 "$SCRIPT_DIR/benchmark_both.py"
else
    echo "=== Node $NODE_RANK connecting to $MASTER_ADDR:$MASTER_PORT ==="
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        "$SCRIPT_DIR/benchmark_both.py"
fi
