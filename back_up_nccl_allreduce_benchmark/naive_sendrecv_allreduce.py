"""
Naive AllReduce using batched isend/irecv (no pipelining, no buffer reuse).

Simple two-shot: ReduceScatter + AllGather, each via a single batch_isend_irecv call.
No optimizations — serves as baseline to compare against NCCL allreduce and the
pipelined send/recv version.

Usage:
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python naive_sendrecv_allreduce.py
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python naive_sendrecv_allreduce.py
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


def naive_sendrecv_allreduce(input_tensor):
    """
    Naive two-shot AllReduce with send/recv for world_size=2.

    Step 1 (ReduceScatter):
      - Each rank sends its peer's chunk, receives peer's copy of its own chunk.
      - Local sum to get reduced chunk.
    Step 2 (AllGather):
      - Each rank sends its reduced chunk, receives peer's reduced chunk.
      - Writes result back into input_tensor.
    """
    rank = dist.get_rank()
    peer = 1 - rank
    n = input_tensor.numel()
    half = n // 2

    my_chunk = input_tensor[rank * half: rank * half + half]
    peer_chunk = input_tensor[peer * half: peer * half + half]

    # Step 1: ReduceScatter — exchange and reduce
    recv_buf = torch.empty(half, dtype=input_tensor.dtype, device=input_tensor.device)

    ops = [
        dist.P2POp(dist.isend, peer_chunk.contiguous(), peer),
        dist.P2POp(dist.irecv, recv_buf, peer),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()

    my_reduced = my_chunk + recv_buf  # creates new tensor (no in-place opt)

    # Step 2: AllGather — exchange reduced chunks
    peer_reduced = torch.empty(half, dtype=input_tensor.dtype, device=input_tensor.device)

    ops = [
        dist.P2POp(dist.isend, my_reduced.contiguous(), peer),
        dist.P2POp(dist.irecv, peer_reduced, peer),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()

    # Reconstruct via torch.cat (no zero-copy opt)
    if rank == 0:
        torch.cat([my_reduced, peer_reduced], out=input_tensor)
    else:
        torch.cat([peer_reduced, my_reduced], out=input_tensor)

    return input_tensor
