"""
Ring AllReduce via CUDA IPC — pipelined, no NCCL for data transfer.

Each rank reads only from its LEFT neighbour via IPC.
Pipeline: each chunk split into P pipe segments for overlap across ranks.
GPU-side flag sync — zero CPU barriers in the hot path.

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode ringp2p
"""

import os
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="ring_p2p_allreduce",
    sources=[os.path.join(_dir, "ring_p2p_allreduce_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


class RingP2PAllReduce:
    def __init__(self, max_elements, dtype, device, rank, world_size,
                 num_pipes=32):
        assert dtype == torch.bfloat16
        self.rank = rank
        self.ws = world_size
        self.num_pipes = num_pipes

        dev_id = device.index if hasattr(device, "index") and device.index is not None \
            else torch.cuda.current_device()

        self.buffer = _ext.alloc_buffer(max_elements, dev_id)

        data_h = _ext.get_handle(self.buffer).cuda()
        flag_h = _ext.alloc_flag().cuda()

        data_list = [torch.empty_like(data_h) for _ in range(world_size)]
        flag_list = [torch.empty_like(flag_h) for _ in range(world_size)]
        dist.all_gather(data_list, data_h)
        dist.all_gather(flag_list, flag_h)

        all_data = torch.stack(data_list).cpu()
        all_flag = torch.stack(flag_list).cpu()

        _ext.init(all_data, all_flag, self.buffer, rank, world_size)

    def __call__(self, tensor):
        n = tensor.numel()
        use_buf = tensor.data_ptr() == self.buffer.data_ptr()
        if not use_buf:
            self.buffer[:n].copy_(tensor)

        _ext.allreduce(self.buffer, n, self.num_pipes)

        if not use_buf:
            tensor.copy_(self.buffer[:n])
        return tensor

    def cleanup(self):
        _ext.cleanup()
