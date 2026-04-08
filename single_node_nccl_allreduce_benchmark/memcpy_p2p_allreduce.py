"""
Two-shot AllReduce using cudaMemcpyAsync + local accumulate kernel.

Supports two modes:
  use_host_staging=False → D2D memcpy via IPC (NVLink, same as direct P2P)
  use_host_staging=True  → GPU→host pinned→GPU (genuine PCIe, ~33 GB/s)

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 bench_memcpy_p2p.py
"""

import os
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="memcpy_p2p_allreduce_ext",
    sources=[os.path.join(_dir, "memcpy_p2p_allreduce_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


class MemcpyP2PAllReduce:
    """Two-shot allreduce: cudaMemcpyAsync (D2D or host-staged) + local accum."""

    def __init__(self, max_elements, dtype, device, rank, world_size,
                 use_host_staging=False, num_pipes=16):
        assert dtype == torch.bfloat16, "Only bf16 supported"
        self.rank = rank
        self.world_size = world_size

        dev_id = device.index if hasattr(device, "index") and device.index is not None \
            else torch.cuda.current_device()

        self.buffer = _ext.alloc_p2p_buffer(max_elements, dev_id)
        flag_handle_cpu = _ext.alloc_flag_buffer()
        data_handle_cpu = _ext.get_ipc_handle(self.buffer)

        data_handle = data_handle_cpu.cuda()
        flag_handle = flag_handle_cpu.cuda()

        data_list = [torch.empty_like(data_handle) for _ in range(world_size)]
        flag_list = [torch.empty_like(flag_handle) for _ in range(world_size)]
        dist.all_gather(data_list, data_handle)
        dist.all_gather(flag_list, flag_handle)

        all_data = torch.stack(data_list).cpu()
        all_flag = torch.stack(flag_list).cpu()

        chunk = max_elements // world_size
        # Ensure chunk is divisible by num_pipes
        while chunk % num_pipes != 0 and num_pipes > 1:
            num_pipes //= 2
        _ext.init_p2p(all_data, all_flag, self.buffer, rank, world_size,
                      chunk, use_host_staging, num_pipes)

    def __call__(self, tensor):
        n = tensor.numel()
        use_buf = tensor.data_ptr() == self.buffer.data_ptr()
        if not use_buf:
            self.buffer[:n].copy_(tensor)

        _ext.fused_allreduce(self.buffer, n)

        if not use_buf:
            tensor.copy_(self.buffer[:n])
        return tensor

    def cleanup(self):
        _ext.cleanup_p2p()
