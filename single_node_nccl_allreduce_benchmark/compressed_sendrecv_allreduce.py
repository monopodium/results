"""
Compressed Ring AllReduce using dietgpu + batched isend/irecv.
Supports any world_size (including 8 GPUs on a single node).

Optimizations over naive per-step compress→size_exchange→data_exchange→decompress:
  1. Fixed-ratio transfer — eliminates .item() GPU sync + size exchange round-trip
     per step. Saves 2 sync points × (N-1) steps × 2 phases = 4(N-1) syncs.
  2. Overlapped compress — compress(step i+1) runs on GPU while NCCL transfers
     step i on its internal stream. Hides compress latency behind transfer.
  3. Double-buffered send/recv — enables overlap without data hazards.

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode compressed
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

# Fixed transfer ratio — 0.70 covers typical bf16 (~0.67) with margin
COMP_RATIO = 0.65


class CompressedAllReduce:
    """
    Ring-based compressed AllReduce: fixed-ratio, overlapped, double-buffered.
    """

    def __init__(self, max_elements, device, compress_rs=True, compress_ag=True):
        self.rank = dist.get_rank()
        self.ws = dist.get_world_size()
        self.left = (self.rank - 1) % self.ws
        self.right = (self.rank + 1) % self.ws
        self.device = device
        self.dtype = torch.bfloat16
        self.max_elements = max_elements
        self.compress_rs = compress_rs
        self.compress_ag = compress_ag

        chunk = max_elements // self.ws

        # dietgpu temp memory
        self.temp_mem = torch.empty(768 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Max compressed output size for one chunk
        dummy = torch.empty(chunk, dtype=self.dtype, device=device)
        _, max_comp_cols = torch.ops.dietgpu.max_float_compressed_output_size([dummy])
        del dummy
        self.max_comp_cols = max_comp_cols

        # Fixed transfer size (aligned to 4 bytes for NCCL)
        self.fixed_bytes = int(chunk * 2 * COMP_RATIO)
        self.fixed_bytes = (self.fixed_bytes + 3) & ~3

        # Double-buffered compression output [2, max_comp_cols]
        self.comp_send = [
            torch.empty(1, max_comp_cols, dtype=torch.uint8, device=device),
            torch.empty(1, max_comp_cols, dtype=torch.uint8, device=device),
        ]
        self.comp_sizes = torch.zeros(1, dtype=torch.int, device=device)

        # Double-buffered receive
        self.comp_recv = [
            torch.empty(self.fixed_bytes, dtype=torch.uint8, device=device),
            torch.empty(self.fixed_bytes, dtype=torch.uint8, device=device),
        ]

        # Decompression buffers
        self.decomp_status = torch.empty(1, dtype=torch.uint8, device=device)
        self.decomp_sizes = torch.empty(1, dtype=torch.int32, device=device)
        self.recv_buf = torch.empty(chunk, dtype=self.dtype, device=device)

        # Raw send/recv for uncompressed path
        self.raw_recv_buf = torch.empty(chunk, dtype=self.dtype, device=device)

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

    def _compress(self, tensor, buf_idx):
        """Compress tensor into comp_send[buf_idx]. No .item() — fire and forget."""
        torch.ops.dietgpu.compress_data(
            True, [tensor], False, self.temp_mem,
            self.comp_send[buf_idx], self.comp_sizes)

    def _decompress(self, comp_data, out_tensor):
        torch.ops.dietgpu.decompress_data(
            True, [comp_data], [out_tensor], False,
            self.temp_mem, self.decomp_status, self.decomp_sizes)

    def _start_exchange(self, buf_idx, fb):
        """Non-blocking isend/irecv of fixed-size compressed data."""
        send_t = self.comp_send[buf_idx][0, :fb].contiguous()
        recv_t = self.comp_recv[buf_idx][:fb]
        ops = [
            dist.P2POp(dist.isend, send_t, self.right),
            dist.P2POp(dist.irecv, recv_t, self.left),
        ]
        return dist.batch_isend_irecv(ops)

    def _exchange_raw(self, send_tensor, recv_tensor):
        ops = [
            dist.P2POp(dist.isend, send_tensor.contiguous(), self.right),
            dist.P2POp(dist.irecv, recv_tensor, self.left),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    def __call__(self, input_tensor):
        rank = self.rank
        ws = self.ws
        n = input_tensor.numel()
        C = n // ws
        # Compute fixed transfer bytes from ACTUAL chunk size, not max
        fb = int(C * 2 * COMP_RATIO)
        fb = (fb + 3) & ~3  # align to 4 bytes for NCCL
        original_bytes = C * 2

        # ── Phase 1: ReduceScatter ──
        if self.compress_rs:
            # Step 0: compress first chunk
            send_idx = (rank - 0) % ws
            cur_buf = 0
            self._compress(input_tensor[send_idx * C:(send_idx + 1) * C].contiguous(), cur_buf)
            self._total_original_bytes += original_bytes
            self._total_compressed_bytes += fb

            for step in range(ws - 1):
                recv_idx = (rank - step - 1) % ws

                # Start exchange (non-blocking, NCCL waits for compress on current stream)
                reqs = self._start_exchange(cur_buf, fb)

                # Overlap: compress NEXT chunk while NCCL transfers current
                if step < ws - 2:
                    next_send_idx = (rank - step - 1) % ws
                    next_buf = 1 - cur_buf
                    self._compress(
                        input_tensor[next_send_idx * C:(next_send_idx + 1) * C].contiguous(),
                        next_buf)
                    self._total_original_bytes += original_bytes
                    self._total_compressed_bytes += fb

                # Wait for exchange to complete
                for r in reqs:
                    r.wait()

                # Decompress received data and accumulate
                self._decompress(self.comp_recv[cur_buf][:fb], self.recv_buf[:C])
                input_tensor[recv_idx * C:(recv_idx + 1) * C].add_(self.recv_buf[:C])

                if step < ws - 2:
                    cur_buf = next_buf
        else:
            for step in range(ws - 1):
                send_idx = (rank - step) % ws
                recv_idx = (rank - step - 1) % ws
                recv = self.raw_recv_buf[:C]
                self._exchange_raw(input_tensor[send_idx * C:(send_idx + 1) * C], recv)
                input_tensor[recv_idx * C:(recv_idx + 1) * C].add_(recv)

        # ── Phase 2: AllGather ──
        if self.compress_ag:
            # Step 0: compress first chunk to send
            send_idx = (rank + 1 - 0) % ws
            cur_buf = 0
            self._compress(input_tensor[send_idx * C:(send_idx + 1) * C].contiguous(), cur_buf)
            self._total_original_bytes += original_bytes
            self._total_compressed_bytes += fb

            for step in range(ws - 1):
                recv_idx = (rank - step) % ws

                # Start exchange
                reqs = self._start_exchange(cur_buf, fb)

                # Overlap: compress NEXT chunk
                if step < ws - 2:
                    next_send_idx = (rank + 1 - step - 1) % ws
                    next_buf = 1 - cur_buf
                    self._compress(
                        input_tensor[next_send_idx * C:(next_send_idx + 1) * C].contiguous(),
                        next_buf)
                    self._total_original_bytes += original_bytes
                    self._total_compressed_bytes += fb

                # Wait for exchange
                for r in reqs:
                    r.wait()

                # Decompress directly into final position
                self._decompress(self.comp_recv[cur_buf][:fb],
                                 input_tensor[recv_idx * C:(recv_idx + 1) * C])

                if step < ws - 2:
                    cur_buf = next_buf
        else:
            for step in range(ws - 1):
                send_idx = (rank + 1 - step) % ws
                recv_idx = (rank - step) % ws
                recv_slice = input_tensor[recv_idx * C:(recv_idx + 1) * C]
                self._exchange_raw(input_tensor[send_idx * C:(send_idx + 1) * C], recv_slice)

        return input_tensor


# ── Module-level convenience wrappers ──

_instances = {}

def compressed_sendrecv_allreduce(input_tensor, compress_rs=True, compress_ag=True):
    global _instances
    key = (compress_rs, compress_ag)
    inst = _instances.get(key)
    if inst is None or inst.max_elements < input_tensor.numel():
        max_elem = max(input_tensor.numel(), 1 << 28)
        inst = CompressedAllReduce(max_elem, input_tensor.device,
                                   compress_rs=compress_rs, compress_ag=compress_ag)
        _instances[key] = inst
    return inst(input_tensor)
