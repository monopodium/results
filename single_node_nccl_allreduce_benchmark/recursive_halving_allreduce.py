"""
Recursive Halving-Doubling AllReduce using isend/irecv.

For power-of-2 world_size, completes in log2(N) steps.
Each step, pairs of ranks exchange and reduce half the data,
halving the active data each time (reduce-scatter), then
double it back (allgather).

For 8 GPUs: only 2 * log2(8) = 6 communication rounds
(vs ring's 14 or naive's 2 rounds with N-1 peers each).

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode rhd
"""

import torch
import torch.distributed as dist


class RecursiveHalvingDoublingAllReduce:
    def __init__(self, max_elements, dtype, device):
        # Recv buffer sized for the largest possible exchange (half the data)
        self.recv_buf = torch.empty(max_elements // 2, dtype=dtype, device=device)

    def __call__(self, input_tensor):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        n = input_tensor.numel()
        recv_buf = self.recv_buf

        # ── Phase 1: Recursive Halving (ReduceScatter) ──
        # At each step k (0..log2(N)-1):
        #   - pair with rank XOR (1 << k)
        #   - exchange half of current working region
        #   - keep the half we're responsible for, reduce with received data
        num_steps = 0
        ws = world_size
        while ws > 1:
            num_steps += 1
            ws >>= 1

        # Track which slice of the original tensor this rank is working on
        slice_start = 0
        slice_size = n

        for step in range(num_steps):
            peer = rank ^ (1 << step)
            half = slice_size // 2

            # Which half do I keep? Determined by bit position
            # If my bit at this position is 0, I keep the lower half and send upper
            # If my bit at this position is 1, I keep the upper half and send lower
            keep_lower = ((rank >> step) & 1) == 0

            if keep_lower:
                send_data = input_tensor[slice_start + half: slice_start + slice_size]
                recv_region = slice_start
            else:
                send_data = input_tensor[slice_start: slice_start + half]
                recv_region = slice_start + half

            rb = recv_buf[:half]
            ops = [
                dist.P2POp(dist.isend, send_data.contiguous(), peer),
                dist.P2POp(dist.irecv, rb, peer),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

            # Reduce: add received data to my kept half
            input_tensor[recv_region: recv_region + half] += rb

            # Narrow working region to the half I kept
            slice_start = recv_region
            slice_size = half

        # Now each rank holds one fully-reduced chunk at [slice_start:slice_start+slice_size]

        # ── Phase 2: Recursive Doubling (AllGather) ──
        # Reverse the process: at each step, exchange with the same peer
        # but now send our reduced chunk and receive theirs
        for step in range(num_steps - 1, -1, -1):
            peer = rank ^ (1 << step)
            half = slice_size  # current slice is one half

            keep_lower = ((rank >> step) & 1) == 0

            if keep_lower:
                # I have the lower half, peer has the upper half
                send_data = input_tensor[slice_start: slice_start + half]
                recv_start = slice_start + half
            else:
                # I have the upper half, peer has the lower half
                send_data = input_tensor[slice_start: slice_start + half]
                recv_start = slice_start - half

            rb = recv_buf[:half]
            ops = [
                dist.P2POp(dist.isend, send_data.contiguous(), peer),
                dist.P2POp(dist.irecv, rb, peer),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

            input_tensor[recv_start: recv_start + half].copy_(rb)

            # Expand working region
            slice_start = min(slice_start, recv_start)
            slice_size = half * 2

        return input_tensor


_rhd_instance = None


def rhd_sendrecv_allreduce(input_tensor):
    global _rhd_instance
    n = input_tensor.numel()
    if _rhd_instance is None or _rhd_instance.recv_buf.numel() < n // 2:
        _rhd_instance = RecursiveHalvingDoublingAllReduce(
            n, input_tensor.dtype, input_tensor.device)
    return _rhd_instance(input_tensor)
