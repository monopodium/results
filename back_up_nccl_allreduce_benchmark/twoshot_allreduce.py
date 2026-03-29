"""
Optimized Two-shot AllReduce using send/recv across 2 nodes.

Key: uses batch_isend_irecv to batch P2P ops into a single NCCL group call,
avoiding the "unbatched P2P" warning and the SIGSEGV from separate communicators.

Optimizations:
  1. batch_isend_irecv: pairs send+recv in one NCCL group call (required for NCCL P2P).
  2. Pre-allocated buffers: recv buffers allocated once and reused.
  3. Zero-copy: operate directly on input_tensor slices, no torch.cat.
  4. Pipelining: split data into K sub-chunks for large messages.

Usage:
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python twoshot_allreduce.py
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python twoshot_allreduce.py
"""

import os
import torch
import torch.distributed as dist


def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


class TwoShotAllReduce:
    """
    Pipelined two-shot AllReduce using batched send/recv for world_size=2.

    Phase 1 (ReduceScatter): batch(send peer_chunk, recv my_chunk) → local add.
    Phase 2 (AllGather):     batch(send my_reduced, recv peer_reduced).

    Data is split into K pipeline stages for large messages.
    """

    def __init__(self, max_elements, dtype, device, num_pipelines=8):
        self.rank = dist.get_rank()
        self.peer = 1 - self.rank
        self.num_pipelines = num_pipelines
        self.dtype = dtype
        self.device = device

        # Pre-allocate worst-case recv buffers
        max_chunk = max_elements // 2
        max_sub = (max_chunk + num_pipelines - 1) // num_pipelines
        self.rs_recv_buf = torch.empty(max_sub, dtype=dtype, device=device)
        self.ag_recv_buf = torch.empty(max_sub, dtype=dtype, device=device)

    def __call__(self, input_tensor):
        rank = self.rank
        peer = self.peer
        n = input_tensor.numel()
        half = n // 2

        my_start = rank * half
        peer_start = peer * half
        my_half = input_tensor[my_start: my_start + half]
        peer_half = input_tensor[peer_start: peer_start + half]

        K = self.num_pipelines
        sub_size = (half + K - 1) // K

        # ── Phase 1: ReduceScatter ──
        for i in range(K):
            lo = i * sub_size
            hi = min(lo + sub_size, half)
            if lo >= hi:
                break
            seg_len = hi - lo

            my_seg = my_half[lo:hi]
            peer_seg = peer_half[lo:hi]
            rs_buf = self.rs_recv_buf[:seg_len]

            # Batch send+recv into one NCCL group call
            ops = [
                dist.P2POp(dist.isend, peer_seg, peer),
                dist.P2POp(dist.irecv, rs_buf, peer),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

            # In-place reduce
            my_seg.add_(rs_buf)

        # ── Phase 2: AllGather ──
        for i in range(K):
            lo = i * sub_size
            hi = min(lo + sub_size, half)
            if lo >= hi:
                break
            seg_len = hi - lo

            my_seg = my_half[lo:hi]
            ag_buf = self.ag_recv_buf[:seg_len]

            ops = [
                dist.P2POp(dist.isend, my_seg, peer),
                dist.P2POp(dist.irecv, ag_buf, peer),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

            # Write received chunk into peer's position in input_tensor
            input_tensor[peer_start + lo: peer_start + hi].copy_(ag_buf)

        return input_tensor


def twoshot_allreduce(input_tensor, world_size, _state={}):
    """Wrapper that lazily creates the TwoShotAllReduce object."""
    key = (input_tensor.device, input_tensor.dtype)
    if key not in _state or _state[key].rs_recv_buf.numel() < input_tensor.numel() // 2:
        max_elem = max(input_tensor.numel(), 1 << 28)
        _state[key] = TwoShotAllReduce(
            max_elem, input_tensor.dtype, input_tensor.device, num_pipelines=8
        )
    return _state[key](input_tensor)


def benchmark_twoshot(size_bytes, world_size, num_warmup=5, num_iters=20):
    """Benchmark two-shot AllReduce for a given buffer size."""
    num_elements = size_bytes // 4
    num_elements = (num_elements // world_size) * world_size
    tensor = torch.ones(num_elements, dtype=torch.float32, device="cuda")

    for _ in range(num_warmup):
        tensor.fill_(1.0)
        twoshot_allreduce(tensor, world_size)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        tensor.fill_(1.0)
        twoshot_allreduce(tensor, world_size)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    actual_bytes = num_elements * 4
    bandwidth_gbps = (actual_bytes * 2) / (elapsed_ms / 1000) / 1e9
    return elapsed_ms, bandwidth_gbps


def verify_correctness(world_size, rank):
    """Verify that two-shot AllReduce produces correct results."""
    for n in [1024, 1 << 20, 1 << 24]:
        tensor = torch.full((n,), float(rank + 1), dtype=torch.float32, device="cuda")
        twoshot_allreduce(tensor, world_size)
        expected = sum(range(1, world_size + 1))
        ok = torch.allclose(tensor, torch.full_like(tensor, expected))
        if rank == 0:
            status = "PASSED" if ok else "FAILED"
            print(f"[Correctness n={n}] {status} (expected {expected}, got {tensor[0].item()})")
        if not ok:
            return False
    return True


def main():
    rank, world_size, _ = setup()

    if not verify_correctness(world_size, rank):
        dist.destroy_process_group()
        return

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
        print(f"\n{'Size':>10s} | {'Latency (ms)':>14s} | {'BusBW (GB/s)':>14s}")
        print("-" * 48)

    for size_bytes, label in sizes:
        latency_ms, busbw = benchmark_twoshot(size_bytes, world_size)
        if rank == 0:
            print(f"{label:>10s} | {latency_ms:>14.3f} | {busbw:>14.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
