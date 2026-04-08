"""
Benchmark: memcpy-based P2P allreduce.
  D2D mode  = cudaMemcpyAsync Device-to-Device via IPC (NVLink)
  Host mode = GPU → host pinned → GPU (genuine PCIe path)

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 bench_memcpy_p2p.py
"""

import os, torch, torch.distributed as dist


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def bench(ar, buf, n, num_warmup=5, num_iters=50):
    for _ in range(num_warmup):
        buf[:n].uniform_(-1, 1)
        ar(buf[:n])
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_iters):
        buf[:n].uniform_(-1, 1)
        ar(buf[:n])
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / num_iters


def main():
    rank, ws = setup()
    from memcpy_p2p_allreduce import MemcpyP2PAllReduce

    sizes = [
        (1 << 22, "4 MB"),
        (1 << 24, "16 MB"),
        (1 << 26, "64 MB"),
        (1 << 28, "256 MB"),
        (1 << 30, "1 GB"),
    ]

    factor = 2.0 * (ws - 1) / ws
    if rank == 0:
        print(f"MemcpyAsync P2P AllReduce — {ws} GPUs, bf16")
        print(f"  D2D mode  = cudaMemcpyAsync D2D via IPC (NVLink)")
        print(f"  Host mode = GPU->host pinned->GPU (PCIe)")
        print()
        print(f"{'Size':>8s}  {'D2D (ms)':>10s}  {'BusBW':>10s}  "
              f"{'Host (ms)':>10s}  {'BusBW':>10s}  {'D2D/Host':>8s}")
        print("-" * 68)

    for nbytes, label in sizes:
        num_elements = nbytes // 2
        num_elements = (num_elements // ws) * ws
        actual_bytes = num_elements * 2

        # D2D mode (global state only allows one instance at a time)
        ar = MemcpyP2PAllReduce(num_elements, torch.bfloat16,
                                torch.device("cuda"), rank, ws,
                                use_host_staging=False)
        ms_d2d = bench(ar, ar.buffer, num_elements)
        ar.cleanup()

        # Host-staged: try several pipe counts, pick the best
        chunk_bytes = (num_elements // ws) * 2
        best_ms, best_np = 1e9, 1
        for np in [1, 2, 4, 8, 16]:
            pipe_bytes = chunk_bytes // np
            if pipe_bytes < (512 << 10):  # skip if pipe < 512KB
                continue
            ar = MemcpyP2PAllReduce(num_elements, torch.bfloat16,
                                    torch.device("cuda"), rank, ws,
                                    use_host_staging=True, num_pipes=np)
            ms = bench(ar, ar.buffer, num_elements, num_warmup=3, num_iters=20)
            ar.cleanup()
            if ms < best_ms:
                best_ms, best_np = ms, np
        ms_host = best_ms

        if rank == 0:
            bw_d2d  = actual_bytes / (ms_d2d  / 1000) / 1e9 * factor
            bw_host = actual_bytes / (ms_host / 1000) / 1e9 * factor
            speedup = ms_host / ms_d2d if ms_d2d > 0 else 0
            print(f"{label:>8s}  {ms_d2d:>8.3f}ms  {bw_d2d:>7.1f} GB  "
                  f"{ms_host:>8.3f}ms  {bw_host:>7.1f} GB  "
                  f"{speedup:>7.1f}x  pipes={best_np}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
