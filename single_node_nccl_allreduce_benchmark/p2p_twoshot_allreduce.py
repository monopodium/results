"""
Custom CUDA two-shot AllReduce via IPC peer-to-peer memory access.
No NCCL for data transfer — GPU kernels read remote GPU memory directly.

v2 optimisations:
  - GPU-side atomic flag barrier (replaces torch.cuda.synchronize + dist.barrier)
  - RS → barrier → AG → barrier all enqueued on GPU stream, zero CPU sync

Uses torch.distributed only for exchanging IPC handles (one-time setup).

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode p2p
"""

import os
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="cuda_p2p_allreduce",
    sources=[os.path.join(_dir, "cuda_p2p_allreduce_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


class P2PTwoShotAllReduce:
    """Two-shot allreduce via CUDA IPC P2P memory access."""

    def __init__(self, max_elements, dtype, device, rank, world_size):
        assert dtype == torch.bfloat16, "Only bf16 supported"
        self.rank = rank
        self.world_size = world_size

        dev_id = device.index if hasattr(device, "index") and device.index is not None \
            else torch.cuda.current_device()

        # Allocate IPC-safe data buffer
        self.buffer = _ext.alloc_p2p_buffer(max_elements, dev_id)

        # Allocate IPC-safe flag buffer (for GPU-side barrier)
        flag_handle_cpu = _ext.alloc_flag_buffer()          # [64] uint8, CPU

        # Get data IPC handle
        data_handle_cpu = _ext.get_ipc_handle(self.buffer)  # [64] uint8, CPU

        # Exchange handles (NCCL backend needs CUDA tensors)
        data_handle = data_handle_cpu.cuda()
        flag_handle = flag_handle_cpu.cuda()

        data_list = [torch.empty_like(data_handle) for _ in range(world_size)]
        flag_list = [torch.empty_like(flag_handle) for _ in range(world_size)]
        dist.all_gather(data_list, data_handle)
        dist.all_gather(flag_list, flag_handle)

        all_data = torch.stack(data_list).cpu()   # [ws, 64]
        all_flag = torch.stack(flag_list).cpu()    # [ws, 64]

        # Open remote memory + flags, build device pointer arrays
        _ext.init_p2p(all_data, all_flag, self.buffer, rank, world_size)

    def __call__(self, tensor):
        n = tensor.numel()

        use_buf = tensor.data_ptr() == self.buffer.data_ptr()
        if not use_buf:
            self.buffer[:n].copy_(tensor)

        # RS → GPU barrier → AG → GPU barrier  (all on GPU, no CPU sync)
        _ext.fused_allreduce(self.buffer, n)

        if not use_buf:
            tensor.copy_(self.buffer[:n])
        return tensor

    def cleanup(self):
        _ext.cleanup_p2p()


_p2p_instance = None


def p2p_twoshot_allreduce(input_tensor):
    """Module-level convenience function with lazy singleton."""
    global _p2p_instance
    n = input_tensor.numel()
    ws = dist.get_world_size()

    if _p2p_instance is None or _p2p_instance.buffer.numel() < n:
        if _p2p_instance is not None:
            _p2p_instance.cleanup()
        _p2p_instance = P2PTwoShotAllReduce(
            n, input_tensor.dtype, input_tensor.device,
            dist.get_rank(), ws,
        )
    return _p2p_instance(input_tensor)
