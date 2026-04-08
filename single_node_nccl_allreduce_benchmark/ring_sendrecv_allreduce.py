"""
Ring AllReduce using isend/irecv.

Two-phase ring algorithm:
  Phase 1 (ReduceScatter): N-1 steps around the ring, accumulating partial sums.
  Phase 2 (AllGather):     N-1 steps, propagating fully-reduced chunks.

Optimizations vs. naive ring:
  * Phase 2 receives directly into the destination chunk — eliminates one full
    chunk copy per step (N-1 copies total, significant when comm is slow).
  * Chunk slice views precomputed once per call (avoids 2*(N-1) view allocations).

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode ring
"""

import torch
import torch.distributed as dist


class RingAllReduce:
    def __init__(self, max_elements, dtype, device, world_size):
        chunk_size = max_elements // world_size
        # Single recv buffer for Phase 1 only (Phase 2 recvs in-place).
        self.recv_buf = torch.empty(chunk_size, dtype=dtype, device=device)

    def __call__(self, input_tensor):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        n = input_tensor.numel()
        C = n // world_size
        left = (rank - 1) % world_size
        right = (rank + 1) % world_size
        recv_buf = self.recv_buf[:C]

        # Precompute chunk views once — slices of a 1D contiguous tensor are
        # already contiguous, and reusing views avoids repeated Python/C++
        # overhead inside the hot loop.
        chunks = [input_tensor.narrow(0, i * C, C) for i in range(world_size)]

        isend = dist.isend
        irecv = dist.irecv
        P2POp = dist.P2POp
        batch = dist.batch_isend_irecv

        # Phase 1: ReduceScatter
        # Step s: send chunk (rank - s) to right, recv from left, add into
        # chunk (rank - s - 1).  Each step's send depends on the previous
        # step's add, so a single recv buffer is sufficient.
        si = rank
        ri = (rank - 1) % world_size
        for _ in range(world_size - 1):
            reqs = batch([
                P2POp(isend, chunks[si], right),
                P2POp(irecv, recv_buf, left),
            ])
            for r in reqs:
                r.wait()
            chunks[ri].add_(recv_buf)
            si = ri
            ri = (ri - 1) % world_size

        # Phase 2: AllGather
        # Step s: send chunk (rank + 1 - s), recv chunk (rank - s) from left.
        # We can receive *directly* into the destination slice — no staging
        # buffer, no copy. This is the main optimization over the naive
        # version, which recv'd into recv_buf and then copy_()'d out.
        si = (rank + 1) % world_size
        ri = rank
        for _ in range(world_size - 1):
            reqs = batch([
                P2POp(isend, chunks[si], right),
                P2POp(irecv, chunks[ri], left),
            ])
            for r in reqs:
                r.wait()
            si = ri
            ri = (ri - 1) % world_size

        return input_tensor


_ring_instance = None


def ring_sendrecv_allreduce(input_tensor):
    global _ring_instance
    n = input_tensor.numel()
    ws = dist.get_world_size()
    if _ring_instance is None or _ring_instance.recv_buf.numel() < n // ws:
        _ring_instance = RingAllReduce(n, input_tensor.dtype, input_tensor.device, ws)
    return _ring_instance(input_tensor)
