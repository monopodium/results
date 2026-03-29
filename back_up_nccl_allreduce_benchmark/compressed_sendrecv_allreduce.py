"""
Compressed AllReduce using dietgpu + batched isend/irecv across 2 nodes.
Uses bfloat16 as the data type. All buffers are pre-allocated and reused.

Supports configurable compression placement:
  compress_rs: compress/decompress around ReduceScatter send/recv
  compress_ag: compress/decompress around AllGather send/recv

Usage:
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python compressed_sendrecv_allreduce.py
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python compressed_sendrecv_allreduce.py
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


def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


class CompressedAllReduce:
    """
    Pre-allocated compressed AllReduce using dietgpu + send/recv for world_size=2.

    Args:
        max_elements: max total elements in input tensor
        device: CUDA device
        compress_rs: whether to compress ReduceScatter communication
        compress_ag: whether to compress AllGather communication
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

        # Temp memory for dietgpu compress/decompress
        self.temp_mem = torch.empty([768 * 1024 * 1024], dtype=torch.uint8, device=device)

        # Figure out max compressed output size for a half-chunk
        dummy = torch.empty(half, dtype=self.dtype, device=device)
        _, max_comp_cols = torch.ops.dietgpu.max_float_compressed_output_size([dummy])

        # Compression output buffers (reused)
        self.comp_send_buf = torch.empty([1, max_comp_cols], dtype=torch.uint8, device=device)
        self.comp_recv_buf = torch.empty([max_comp_cols], dtype=torch.uint8, device=device)
        self.comp_sizes = torch.zeros([1], dtype=torch.int, device=device)
        self.max_comp_cols = max_comp_cols

        # Decompression status/sizes (reused)
        self.decomp_status = torch.empty([1], dtype=torch.uint8, device=device)
        self.decomp_sizes = torch.empty([1], dtype=torch.int32, device=device)

        # Size exchange buffers (reused)
        self.send_size_t = torch.empty([1], dtype=torch.int64, device=device)
        self.recv_size_t = torch.empty([1], dtype=torch.int64, device=device)

        # Recv decompressed buffer (reused)
        self.recv_buf = torch.empty(half, dtype=self.dtype, device=device)

        # Peer reduced buffer for allgather step (reused)
        self.peer_reduced = torch.empty(half, dtype=self.dtype, device=device)

        # Raw send/recv buffer for uncompressed path (reused)
        self.raw_recv_buf = torch.empty(half, dtype=self.dtype, device=device)

        # Compression ratio tracking (not counted in timing)
        self._total_original_bytes = 0
        self._total_compressed_bytes = 0

    def reset_stats(self):
        """Reset compression ratio counters."""
        self._total_original_bytes = 0
        self._total_compressed_bytes = 0

    def get_comp_ratio(self):
        """Return compression ratio (compressed/original). Lower is better."""
        if self._total_original_bytes == 0:
            return 0.0
        return self._total_compressed_bytes / self._total_original_bytes

    def _compress(self, tensor):
        """Compress tensor into pre-allocated buffers. Returns actual compressed size."""
        self.comp_sizes.zero_()
        _, self.comp_sizes, _ = torch.ops.dietgpu.compress_data(
            True, [tensor], False, self.temp_mem, self.comp_send_buf, self.comp_sizes
        )
        return self.comp_sizes[0].item()

    def _decompress(self, comp_data, out_tensor):
        """Decompress from comp_data into out_tensor."""
        torch.ops.dietgpu.decompress_data(
            True, [comp_data], [out_tensor], False,
            self.temp_mem, self.decomp_status, self.decomp_sizes
        )

    def _exchange_compressed(self, send_size):
        """Exchange compressed sizes first, then compressed data."""
        peer = self.peer

        # Exchange sizes
        self.send_size_t[0] = send_size
        ops = [
            dist.P2POp(dist.isend, self.send_size_t, peer),
            dist.P2POp(dist.irecv, self.recv_size_t, peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

        peer_comp_size = self.recv_size_t.item()

        # Exchange compressed data (only actual compressed bytes)
        comp_recv = self.comp_recv_buf[:peer_comp_size]
        comp_send = self.comp_send_buf[0, :send_size].contiguous()

        ops = [
            dist.P2POp(dist.isend, comp_send, peer),
            dist.P2POp(dist.irecv, comp_recv, peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

        return comp_recv

    def _exchange_raw(self, send_tensor, recv_tensor):
        """Exchange raw (uncompressed) tensors."""
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

        original_bytes = half * 2  # bfloat16 = 2 bytes

        # ── Step 1: ReduceScatter ──
        if self.compress_rs:
            send_size = self._compress(peer_chunk.contiguous())
            self._total_original_bytes += original_bytes
            self._total_compressed_bytes += send_size
            comp_recv = self._exchange_compressed(send_size)
            self._decompress(comp_recv, self.recv_buf[:half])
            my_chunk.add_(self.recv_buf[:half])
        else:
            recv = self.raw_recv_buf[:half]
            self._exchange_raw(peer_chunk, recv)
            my_chunk.add_(recv)

        # ── Step 2: AllGather ──
        if self.compress_ag:
            send_size = self._compress(my_chunk.contiguous())
            self._total_original_bytes += original_bytes
            self._total_compressed_bytes += send_size
            comp_recv = self._exchange_compressed(send_size)
            self._decompress(comp_recv, self.peer_reduced[:half])
            input_tensor[peer_start: peer_start + half].copy_(self.peer_reduced[:half])
        else:
            recv = self.raw_recv_buf[:half]
            self._exchange_raw(my_chunk, recv)
            input_tensor[peer_start: peer_start + half].copy_(recv)

        return input_tensor


# ── Module-level convenience wrappers ──

_instances = {}

def compressed_sendrecv_allreduce(input_tensor, compress_rs=True, compress_ag=True):
    """
    Compressed AllReduce with pre-allocated buffers.

    Args:
        compress_rs: compress ReduceScatter communication
        compress_ag: compress AllGather communication
    """
    global _instances
    key = (compress_rs, compress_ag)
    inst = _instances.get(key)
    if inst is None or inst.max_elements < input_tensor.numel():
        max_elem = max(input_tensor.numel(), 1 << 28)
        inst = CompressedAllReduce(max_elem, input_tensor.device,
                                   compress_rs=compress_rs, compress_ag=compress_ag)
        _instances[key] = inst
    return inst(input_tensor)
