"""
Compressed AllReduce using dietgpu + batched isend/irecv across 2 nodes.
NO size exchange variant: assumes compressed size = 0.65 * original size.
This eliminates the size-exchange round-trip and the .item() GPU sync.

Usage:
  torchrun ... benchmark_both.py --mode compressed_nosize
"""

import os
import glob as _glob
import torch
import torch.distributed as dist

# ── Load dietgpu library ──
_so_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "uccl", "thirdparty", "dietgpu", "build")
_search_dirs = [
    _so_dir,
    "/home/ubuntu/efs/shuangma/uccl/thirdparty/dietgpu/build",
]
_so_files = []
for d in _search_dirs:
    _so_files = _glob.glob(os.path.join(d, "lib.*", "p2p_dietgpu*.so"))
    if _so_files:
        break
if not _so_files:
    raise RuntimeError(f"Could not find p2p_dietgpu*.so. Searched: {_search_dirs}")
torch.ops.load_library(_so_files[0])

# Fixed compression ratio assumption (compressed / original)
COMP_RATIO = 0.65


class CompressedNosizeAllReduce:
    """
    Pre-allocated compressed AllReduce for world_size=2.
    No size exchange — uses fixed assumed compressed size (0.65 * original).
    Eliminates the size-exchange round-trip and .item() GPU sync.
    """

    def __init__(self, max_elements, device, compress_rs=True, compress_ag=True):
        self.rank = dist.get_rank()
        self.peer = 1 - self.rank
        self.device = device
        self.dtype = torch.bfloat16
        self.max_elements = max_elements
        self.compress_rs = compress_rs
        self.compress_ag = compress_ag

        half = max_elements // 2

        # Small tensor for cross-node sync (all_reduce is more reliable than barrier)
        self._sync_tensor = torch.zeros(1, dtype=torch.int32, device=device)

        # Temp memory for dietgpu compress/decompress
        self.temp_mem = torch.empty([768 * 1024 * 1024], dtype=torch.uint8, device=device)

        # Max compressed output size
        dummy = torch.empty(half, dtype=self.dtype, device=device)
        _, max_comp_cols = torch.ops.dietgpu.max_float_compressed_output_size([dummy])

        # Compression output buffers (reused)
        self.comp_send_buf = torch.empty([1, max_comp_cols], dtype=torch.uint8, device=device)
        self.comp_recv_buf = torch.empty([max_comp_cols], dtype=torch.uint8, device=device)
        self.comp_sizes = torch.zeros([1], dtype=torch.int, device=device)

        # Decompression status/sizes (reused)
        self.decomp_status = torch.empty([1], dtype=torch.uint8, device=device)
        self.decomp_sizes = torch.empty([1], dtype=torch.int32, device=device)

        # Recv decompressed buffer (reused)
        self.recv_buf = torch.empty(half, dtype=self.dtype, device=device)

        # Peer reduced buffer for allgather step (reused)
        self.peer_reduced = torch.empty(half, dtype=self.dtype, device=device)

        # Raw send/recv buffer for uncompressed path (reused)
        self.raw_recv_buf = torch.empty(half, dtype=self.dtype, device=device)

        # Stats
        self._total_original_bytes = 0
        self._total_compressed_bytes = 0

    def reset_stats(self):
        self._total_original_bytes = 0
        self._total_compressed_bytes = 0

    def get_comp_ratio(self):
        if self._total_original_bytes == 0:
            return 0.0
        return self._total_compressed_bytes / self._total_original_bytes

    def _compress(self, tensor):
        """Compress tensor. No .item() call — uses fixed size assumption."""
        self.comp_sizes.zero_()
        # compress_data returns (Tensor[], Tensor, int) where Tensor[] is a list
        # of narrowed views. We discard the list to keep comp_send_buf as 2D tensor.
        _, self.comp_sizes, _ = torch.ops.dietgpu.compress_data(
            True, [tensor], False, self.temp_mem, self.comp_send_buf, self.comp_sizes
        )

    def _decompress(self, comp_data, out_tensor):
        torch.ops.dietgpu.decompress_data(
            True, [comp_data], [out_tensor], False,
            self.temp_mem, self.decomp_status, self.decomp_sizes
        )

    def _exchange_compressed_fixed(self, half):
        """Exchange compressed data using fixed assumed size. No size exchange."""
        peer = self.peer
        fixed_bytes = int(half * 2 * COMP_RATIO)  # bf16 = 2 bytes/element
        # Round up to multiple of 4 bytes for NCCL alignment
        fixed_bytes = (fixed_bytes + 3) & ~3

        comp_send = self.comp_send_buf[0, :fixed_bytes].contiguous()
        comp_recv = self.comp_recv_buf[:fixed_bytes]

        ops = [
            dist.P2POp(dist.isend, comp_send, peer),
            dist.P2POp(dist.irecv, comp_recv, peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

        return comp_recv

    def _exchange_raw(self, send_tensor, recv_tensor):
        peer = self.peer
        ops = [
            dist.P2POp(dist.isend, send_tensor.contiguous(), peer),
            dist.P2POp(dist.irecv, recv_tensor, peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    def __call__(self, input_tensor):
        rank = self.rank
        n = input_tensor.numel()
        half = n // 2

        my_start = rank * half
        peer_start = (1 - rank) * half
        my_chunk = input_tensor[my_start: my_start + half]
        peer_chunk = input_tensor[peer_start: peer_start + half]

        original_bytes = half * 2

        # ── Step 1: ReduceScatter ──
        if self.compress_rs:
            self._compress(peer_chunk.contiguous())
            self._total_original_bytes += original_bytes
            self._total_compressed_bytes += int(original_bytes * COMP_RATIO)
            comp_recv = self._exchange_compressed_fixed(half)
            self._decompress(comp_recv, self.recv_buf[:half])
            my_chunk.add_(self.recv_buf[:half])
        else:
            recv = self.raw_recv_buf[:half]
            self._exchange_raw(peer_chunk, recv)
            my_chunk.add_(recv)

        # ── Step 2: AllGather ──
        if self.compress_ag:
            self._compress(my_chunk.contiguous())
            self._total_original_bytes += original_bytes
            self._total_compressed_bytes += int(original_bytes * COMP_RATIO)
            comp_recv = self._exchange_compressed_fixed(half)
            self._decompress(comp_recv, self.peer_reduced[:half])
            input_tensor[peer_start: peer_start + half].copy_(self.peer_reduced[:half])
        else:
            recv = self.raw_recv_buf[:half]
            self._exchange_raw(my_chunk, recv)
            input_tensor[peer_start: peer_start + half].copy_(recv)

        return input_tensor


# ── Module-level convenience wrappers ──

_nosize_instances = {}

def compressed_nosize_allreduce(input_tensor, compress_rs=True, compress_ag=True):
    global _nosize_instances
    key = (compress_rs, compress_ag)
    inst = _nosize_instances.get(key)
    if inst is None or inst.max_elements < input_tensor.numel():
        max_elem = max(input_tensor.numel(), 1 << 28)
        inst = CompressedNosizeAllReduce(max_elem, input_tensor.device,
                                         compress_rs=compress_rs, compress_ag=compress_ag)
        _nosize_instances[key] = inst
    return inst(input_tensor)
