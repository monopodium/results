"""
Standard NCCL AllReduce across 2 nodes.

Usage:
  # Node 0 (master):
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python nccl_standard_allreduce.py

  # Node 1:
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python nccl_standard_allreduce.py

  Or use torchrun:
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<ip> --master_port=29500 nccl_standard_allreduce.py
"""

import os
import time
import torch
import torch.distributed as dist


def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def benchmark_allreduce(size_bytes, world_size, num_warmup=5, num_iters=20):
    """Benchmark standard NCCL AllReduce for a given buffer size."""
    num_elements = size_bytes // 4  # float32 = 4 bytes
    tensor = torch.ones(num_elements, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(num_warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    algbw = size_bytes / (elapsed_ms / 1000) / 1e9
    busbw = algbw * 2.0 * (world_size - 1) / world_size
    return elapsed_ms, algbw, busbw


def main():
    rank, world_size, local_rank = setup()

    sizes = [
        (1 << 10, "1 KB"),
        (1 << 14, "16 KB"),
        (1 << 18, "256 KB"),
        (1 << 20, "1 MB"),
        (1 << 22, "4 MB"),
        (1 << 24, "16 MB"),
        (1 << 26, "64 MB"),
        (1 << 28, "256 MB"),
        (1 << 30, "1 GB"),
    ]

    if rank == 0:
        print(f"{'Size':>10s} | {'Latency (ms)':>14s} | {'AlgBW (GB/s)':>14s} | {'BusBW (GB/s)':>14s}")
        print("-" * 64)

    for size_bytes, label in sizes:
        latency_ms, algbw, busbw = benchmark_allreduce(size_bytes, world_size)
        if rank == 0:
            print(f"{label:>10s} | {latency_ms:>14.3f} | {algbw:>14.2f} | {busbw:>14.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
