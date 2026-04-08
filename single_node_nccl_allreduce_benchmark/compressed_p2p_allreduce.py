"""
Compressed P2P AllReduce: dietgpu compression + CUDA IPC peer-to-peer transfer.

v3: Hybrid RS (uncompressed P2P) + AG (compressed P2P via dietgpu).

RS uses the same raw-read kernel as p2p_twoshot — no compression overhead.
AG compresses the reduced chunk, transfers compressed data via IPC, batch
decompresses on the remote side.

Kernel launch count (8 GPUs):
  RS: 7 rs_one_source + 1 barrier = 8
  AG: 1 compress + 1 barrier + 7 copy + 1 decompress + 1 barrier = 11
  Total: 19 operations

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode comp_p2p
"""

import os
import glob as _glob
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

# ── Load dietgpu library ──
_so_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "uccl", "thirdparty", "dietgpu", "build")
_search_dirs = [
    _so_dir,
    "/home/ubuntu/efs/shuangma/uccl/thirdparty/dietgpu/build",
]
_dg_loaded = False
for d in _search_dirs:
    so_files = _glob.glob(os.path.join(d, "lib.*", "p2p_dietgpu*.so"))
    if so_files:
        torch.ops.load_library(so_files[0])
        _dg_loaded = True
        break
assert _dg_loaded, "dietgpu .so not found"

# ── Load CUDA extension ──
_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="compressed_p2p_kernel",
    sources=[os.path.join(_dir, "compressed_p2p_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

# Fixed transfer ratio for compressed AG
COMP_RATIO = 0.75


class CompressedP2PAllReduce:
    """Hybrid AllReduce: uncompressed P2P RS + compressed P2P AG."""

    def __init__(self, max_elements, dtype, device, rank, world_size):
        assert dtype == torch.bfloat16, "Only bf16 supported"
        self.rank = rank
        self.ws = world_size
        self.device = device
        self.npeers = world_size - 1
        dev_id = device.index if hasattr(device, "index") and device.index is not None \
            else torch.cuda.current_device()

        max_elements = (max_elements // world_size) * world_size
        chunk = max_elements // world_size
        self.chunk = chunk

        # ── IPC data buffer (for uncompressed RS reads) ──
        self.buffer = self._alloc_p2p_buffer(max_elements, dev_id, dtype)
        data_handle = _ext.get_ipc_handle(self.buffer)

        # ── Compressed buffer sizing (for AG) ──
        dummy = torch.empty(chunk, dtype=dtype, device=device)
        _, max_comp_cols = torch.ops.dietgpu.max_float_compressed_output_size([dummy])
        del dummy
        self.max_comp_cols = max_comp_cols
        self.comp_stride = ((max_comp_cols + 255) // 256) * 256

        # Fixed transfer size per chunk
        self.fixed_bytes = int(chunk * 2 * COMP_RATIO)
        self.fixed_bytes = (self.fixed_bytes + 255) & ~255

        # ── IPC compressed buffer: ws slots × comp_stride bytes ──
        comp_buf_bytes = world_size * self.comp_stride
        self.ipc_comp = _ext.alloc_ipc_buffer(comp_buf_bytes, dev_id)
        comp_handle = _ext.get_ipc_handle(self.ipc_comp)

        # ── IPC flag buffer ──
        flag_handle = _ext.alloc_flag_buffer()

        # ── Exchange ALL IPC handles ──
        dh = data_handle.cuda()
        ch = comp_handle.cuda()
        fh = flag_handle.cuda()
        data_list = [torch.empty_like(dh) for _ in range(world_size)]
        comp_list = [torch.empty_like(ch) for _ in range(world_size)]
        flag_list = [torch.empty_like(fh) for _ in range(world_size)]
        dist.all_gather(data_list, dh)
        dist.all_gather(comp_list, ch)
        dist.all_gather(flag_list, fh)
        all_data = torch.stack(data_list).cpu()
        all_comp = torch.stack(comp_list).cpu()
        all_flag = torch.stack(flag_list).cpu()

        _ext.init_comp_p2p(all_data, all_comp, all_flag,
                           self.buffer, self.ipc_comp,
                           rank, world_size, self.comp_stride)

        # ── AG work buffers ──
        self.all_recv = torch.empty(self.npeers, self.comp_stride,
                                    dtype=torch.uint8, device=device)
        self.comp_sizes = torch.empty(1, dtype=torch.int32, device=device)
        self.batch_status = torch.empty(self.npeers, dtype=torch.uint8, device=device)
        self.batch_sizes = torch.empty(self.npeers, dtype=torch.int32, device=device)

        # dietgpu temp memory — only need single-chunk compress so 768MB is enough
        self.temp_mem = torch.empty(768 * 1024 * 1024, dtype=torch.uint8, device=device)

    @staticmethod
    def _alloc_p2p_buffer(numel, dev, dtype):
        """Allocate IPC-safe bf16 buffer via cudaMalloc (same as p2p_twoshot)."""
        raw = _ext.alloc_ipc_buffer(numel * 2, dev)          # uint8
        # Reinterpret the raw uint8 buffer as bf16
        t = raw.view(torch.bfloat16)[:numel]
        t._ipc_raw = raw          # prevent GC of underlying cudaMalloc
        return t

    def __call__(self, tensor):
        n = tensor.numel()
        n = (n // self.ws) * self.ws
        chunk = n // self.ws
        rank = self.rank
        ws = self.ws
        npeers = self.npeers
        fb = self.fixed_bytes

        use_buf = tensor.data_ptr() == self.buffer.data_ptr()
        if not use_buf:
            self.buffer[:n].copy_(tensor)
        data = self.buffer[:n]

        # ════════════════════════════════════════════════
        #  ReduceScatter: raw P2P reads (no compression)
        #  7 rs_one_source kernels + 1 barrier
        # ════════════════════════════════════════════════
        _ext.run_rs(data, n)

        # ════════════════════════════════════════════════
        #  AllGather: compress → barrier → copy → decompress
        # ════════════════════════════════════════════════

        my_data = data[rank * chunk:(rank + 1) * chunk]

        # 1. Compress reduced chunk into slot[rank]
        ag_out = self.ipc_comp[rank * self.comp_stride:
                               (rank + 1) * self.comp_stride].view(1, self.comp_stride)
        torch.ops.dietgpu.compress_data(
            True, [my_data], False, self.temp_mem, ag_out, self.comp_sizes)

        # 2. Barrier — compressed reduced chunks visible
        _ext.gpu_barrier()

        # 3. Launch ALL copies (GPU pipelines NVLink reads)
        for i in range(npeers):
            s = (rank + 1 + i) % ws
            _ext.copy_from_remote(
                self.all_recv[i], s, s * self.comp_stride, fb)

        # 4. Batch decompress directly into final data positions
        decomp_in = [self.all_recv[i:i+1, :self.max_comp_cols]
                     for i in range(npeers)]
        staggered = [(rank + 1 + i) % ws for i in range(npeers)]
        decomp_out = [data[s * chunk:(s + 1) * chunk] for s in staggered]
        torch.ops.dietgpu.decompress_data(
            True, decomp_in, decomp_out, False,
            self.temp_mem, self.batch_status, self.batch_sizes)

        # 5. Barrier — AG complete
        _ext.gpu_barrier()

        if not use_buf:
            tensor.copy_(data)
        return tensor

    def cleanup(self):
        _ext.cleanup_comp_p2p()
