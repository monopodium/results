"""
Two-shot AllReduce using NCCL collective primitives:
  reduce_scatter_tensor + all_gather_into_tensor.

This is the theoretical upper bound for a two-shot approach — both ops
use NCCL's optimized internal algorithms (ring/tree), not isend/irecv.

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode twoshot
"""

import torch
import torch.distributed as dist


class TwoShotCollectiveAllReduce:
    def __init__(self, max_elements, dtype, device, world_size):
        chunk = max_elements // world_size
        self.scatter_out = torch.empty(chunk, dtype=dtype, device=device)

    def __call__(self, input_tensor):
        world_size = dist.get_world_size()
        n = input_tensor.numel()
        chunk = n // world_size
        scatter_out = self.scatter_out[:chunk]

        # Phase 1: ReduceScatter — each rank gets 1/N of the sum
        dist.reduce_scatter_tensor(scatter_out, input_tensor, op=dist.ReduceOp.SUM)

        # Phase 2: AllGather — broadcast each rank's chunk to all
        dist.all_gather_into_tensor(input_tensor, scatter_out)

        return input_tensor


_twoshot_instance = None


def twoshot_collective_allreduce(input_tensor):
    global _twoshot_instance
    n = input_tensor.numel()
    ws = dist.get_world_size()
    if _twoshot_instance is None or _twoshot_instance.scatter_out.numel() < n // ws:
        _twoshot_instance = TwoShotCollectiveAllReduce(
            n, input_tensor.dtype, input_tensor.device, ws)
    return _twoshot_instance(input_tensor)
