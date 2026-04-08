"""
Compressed Two-Shot AllReduce using dietgpu + all-to-all batched isend/irecv.

Supports any world_size >= 2 (tested with 2, 4, 8, 16 GPUs — single-node
or multi-node via torchrun).

Unlike the ring version (compressed_sendrecv_allreduce.py) which has N-1
sequential steps per phase, this version does ONE batch exchange per phase:

  Ring (N=8): 14 sequential (compress→send→recv→decompress) = 28+ serial ops
  Two-shot:   batch_compress → 1 batch_exchange → batch_decompress → fused_sum
              RS + AG = ~7 total ops regardless of N

Key advantages for small messages:
  - Only 2 NCCL batch_isend_irecv calls (vs N-1 in ring)
  - Only 4 dietgpu calls total: 2 batch compress + 2 batch decompress
  - Fused accumulation via torch.sum(dim=0)
  - All peers exchange concurrently (better for socket transport)

Usage:
  # single node, 8 GPUs
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode compressed_nosize
  # 2 nodes × 8 GPUs = 16 ranks
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=<R> \\
           --master_addr=<IP> benchmark_both.py --mode compressed_nosize
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

COMP_RATIO = 0.65


class CompressedNosizeAllReduce:
    """
    Two-shot all-to-all compressed AllReduce.
    Batch compress → batch exchange → batch decompress → fused sum.
    """

    def __init__(self, max_elements, device, compress_rs=True, compress_ag=True):
        self.rank = dist.get_rank()
        self.ws = dist.get_world_size()
        ws = self.ws
        npeers = ws - 1
        self.npeers = npeers
        self.device = device
        self.dtype = torch.bfloat16
        self.max_elements = max_elements
        self.compress_rs = compress_rs
        self.compress_ag = compress_ag

        chunk = max_elements // ws

        # Max compressed output size
        dummy = torch.empty(chunk, dtype=self.dtype, device=device)
        _, max_comp_cols = torch.ops.dietgpu.max_float_compressed_output_size([dummy])
        del dummy
        self.max_comp_cols = max_comp_cols

        # Max fixed transfer bytes
        max_fb = int(chunk * 2 * COMP_RATIO)
        max_fb = (max_fb + 3) & ~3
        self.max_fb = max_fb

        # ── RS buffers ──
        # Batch compress output: [ws, max_comp_cols]
        self.rs_comp_out = torch.empty(ws, max_comp_cols, dtype=torch.uint8, device=device)
        self.rs_comp_sizes = torch.zeros(ws, dtype=torch.int32, device=device)
        # Receive compressed chunks — separate contiguous buffers for NCCL alignment
        self.rs_recv = [torch.empty(max_fb, dtype=torch.uint8, device=device)
                        for _ in range(npeers)]
        # Decompress output (contiguous for fused sum)
        self.rs_decomp = torch.empty(npeers * chunk, dtype=self.dtype, device=device)
        self.rs_decomp_status = torch.empty(npeers, dtype=torch.uint8, device=device)
        self.rs_decomp_sizes = torch.empty(npeers, dtype=torch.int32, device=device)
        # Raw RS recv for uncompressed path
        self.rs_raw_recv = torch.empty(npeers * chunk, dtype=self.dtype, device=device)

        # ── AG buffers ──
        self.ag_comp_out = torch.empty(1, max_comp_cols, dtype=torch.uint8, device=device)
        self.ag_comp_sizes = torch.zeros(1, dtype=torch.int32, device=device)
        self.ag_recv = [torch.empty(max_fb, dtype=torch.uint8, device=device)
                        for _ in range(npeers)]
        self.ag_decomp_status = torch.empty(npeers, dtype=torch.uint8, device=device)
        self.ag_decomp_sizes = torch.empty(npeers, dtype=torch.int32, device=device)

        # dietgpu temp memory — scale with batch size so large world_size
        # (e.g. 16 ranks → 16-way batch compress) always has enough workspace.
        # Rule of thumb: ~3× total-input-bytes of the batch compress, with a
        # 1.5 GB floor and 6 GB cap.
        total_batch_bytes = max_elements * 2  # bf16 = 2 B/elem, batch covers whole input
        temp_bytes = max(1536 * 1024 * 1024, min(6 * 1024 * 1024 * 1024, 3 * total_batch_bytes))
        self.temp_mem = torch.empty(int(temp_bytes), dtype=torch.uint8, device=device)

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

    def __call__(self, input_tensor):
        rank = self.rank
        ws = self.ws
        npeers = self.npeers
        n = input_tensor.numel()
        assert n % ws == 0, f"tensor numel {n} not divisible by world_size {ws}"
        C = n // ws
        fb = int(C * 2 * COMP_RATIO)
        fb = (fb + 3) & ~3
        # Clamp to the preallocated per-row width (safety when called with
        # a smaller tensor than max_elements).
        fb = min(fb, self.max_fb)
        original_bytes = C * 2

        # ════════ ReduceScatter ════════
        if self.compress_rs:
            # 1. Batch compress all ws chunks (1 dietgpu call)
            chunks = [input_tensor[i * C:(i + 1) * C] for i in range(ws)]
            torch.ops.dietgpu.compress_data(
                True, chunks, False, self.temp_mem,
                self.rs_comp_out[:ws], self.rs_comp_sizes[:ws])

            # 2. All-to-all exchange (1 NCCL batch)
            ops = []
            buf_idx = 0
            for peer in range(ws):
                if peer == rank:
                    continue
                # Send compressed chunk[peer] to peer
                ops.append(dist.P2POp(dist.isend,
                           self.rs_comp_out[peer, :fb].contiguous(), peer))
                # Recv compressed chunk[rank] from peer
                ops.append(dist.P2POp(dist.irecv,
                           self.rs_recv[buf_idx][:fb], peer))
                buf_idx += 1
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

            # 3. Batch decompress all received (1 dietgpu call)
            decomp_in = [self.rs_recv[i][:fb].unsqueeze(0) for i in range(npeers)]
            decomp_out = [self.rs_decomp[i * C:(i + 1) * C] for i in range(npeers)]
            torch.ops.dietgpu.decompress_data(
                True, decomp_in, decomp_out, False,
                self.temp_mem, self.rs_decomp_status, self.rs_decomp_sizes)

            # 4. Fused accumulate (1 kernel)
            my_start = rank * C
            input_tensor[my_start:my_start + C] += \
                self.rs_decomp[:npeers * C].view(npeers, C).sum(dim=0)

            self._total_original_bytes += npeers * original_bytes
            self._total_compressed_bytes += npeers * fb
        else:
            # Uncompressed all-to-all RS
            ops = []
            buf_idx = 0
            for peer in range(ws):
                if peer == rank:
                    continue
                ops.append(dist.P2POp(dist.isend,
                           input_tensor[peer * C:(peer + 1) * C], peer))
                ops.append(dist.P2POp(dist.irecv,
                           self.rs_raw_recv[buf_idx * C:(buf_idx + 1) * C], peer))
                buf_idx += 1
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            my_start = rank * C
            input_tensor[my_start:my_start + C] += \
                self.rs_raw_recv[:npeers * C].view(npeers, C).sum(dim=0)

        # ════════ AllGather ════════
        if self.compress_ag:
            # 1. Compress my reduced chunk (1 dietgpu call)
            my_start = rank * C
            my_reduced = input_tensor[my_start:my_start + C]
            torch.ops.dietgpu.compress_data(
                True, [my_reduced], False, self.temp_mem,
                self.ag_comp_out, self.ag_comp_sizes)

            # 2. All-to-all exchange (1 NCCL batch)
            my_comp = self.ag_comp_out[0, :fb].contiguous()
            ops = []
            buf_idx = 0
            for peer in range(ws):
                if peer == rank:
                    continue
                ops.append(dist.P2POp(dist.isend, my_comp, peer))
                ops.append(dist.P2POp(dist.irecv,
                           self.ag_recv[buf_idx][:fb], peer))
                buf_idx += 1
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

            # 3. Batch decompress into final positions (1 dietgpu call)
            decomp_in = [self.ag_recv[i][:fb].unsqueeze(0) for i in range(npeers)]
            peers = [p for p in range(ws) if p != rank]
            decomp_out = [input_tensor[p * C:(p + 1) * C] for p in peers]
            torch.ops.dietgpu.decompress_data(
                True, decomp_in, decomp_out, False,
                self.temp_mem, self.ag_decomp_status, self.ag_decomp_sizes)

            self._total_original_bytes += npeers * original_bytes
            self._total_compressed_bytes += npeers * fb
        else:
            # Uncompressed all-to-all AG (recv directly into input_tensor)
            my_start = rank * C
            my_reduced = input_tensor[my_start:my_start + C]
            ops = []
            for peer in range(ws):
                if peer == rank:
                    continue
                ops.append(dist.P2POp(dist.isend, my_reduced, peer))
                ops.append(dist.P2POp(dist.irecv,
                           input_tensor[peer * C:(peer + 1) * C], peer))
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

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
                                         compress_rs=compress_rs,
                                         compress_ag=compress_ag)
        _nosize_instances[key] = inst
    return inst(input_tensor)
