"""
Benchmark: NCCL AllReduce vs send/recv AllReduce variants.
Reports both CUDA event time (GPU-side) and wall-clock time (includes CPU overhead).

Usage:
  torchrun ... benchmark_both.py --mode naive                    # NCCL vs Naive
  torchrun ... benchmark_both.py --mode ring                     # NCCL vs Ring
  torchrun ... benchmark_both.py --mode compressed               # NCCL vs Compressed (both RS+AG)
  torchrun ... benchmark_both.py --mode compressed --compress rs # Compress ReduceScatter only
  torchrun ... benchmark_both.py --mode compressed --compress ag # Compress AllGather only
  torchrun ... benchmark_both.py --mode compressed_nosize        # No size exchange (fixed 0.65 ratio)
  torchrun ... benchmark_both.py --mode all                      # All variants
"""

import argparse
import csv
import os
import time
import torch
import torch.distributed as dist

from naive_sendrecv_allreduce import naive_sendrecv_allreduce
from ring_sendrecv_allreduce import ring_sendrecv_allreduce
from recursive_halving_allreduce import rhd_sendrecv_allreduce
from twoshot_collective_allreduce import twoshot_collective_allreduce


def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# ─── Common ──────────────────────────────────────────────────────────

DTYPE = torch.bfloat16

def make_rand_tensor(num_elements):
    """Create a random bfloat16 tensor uniformly distributed in [-1, 1]."""
    return torch.empty(num_elements, dtype=torch.float32, device="cuda").uniform_(-1.0, 1.0).to(DTYPE)

def reset_rand(tensor):
    """Re-fill tensor with fresh random data in [-1, 1]."""
    tensor.copy_(torch.empty_like(tensor, dtype=torch.float32).uniform_(-1.0, 1.0))


# ─── Bench helpers (return cuda_ms, wall_ms) ─────────────────────────

def bench_standard(num_elements, num_warmup=5, num_iters=50):
    tensor = make_rand_tensor(num_elements)
    for _ in range(num_warmup):
        reset_rand(tensor)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_naive(num_elements, world_size, num_warmup=5, num_iters=50):
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)
    for _ in range(num_warmup):
        reset_rand(tensor)
        naive_sendrecv_allreduce(tensor)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        naive_sendrecv_allreduce(tensor)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_ring(num_elements, world_size, num_warmup=5, num_iters=50):
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)
    for _ in range(num_warmup):
        reset_rand(tensor)
        ring_sendrecv_allreduce(tensor)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        ring_sendrecv_allreduce(tensor)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_rhd(num_elements, world_size, num_warmup=5, num_iters=50):
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)
    for _ in range(num_warmup):
        reset_rand(tensor)
        rhd_sendrecv_allreduce(tensor)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        rhd_sendrecv_allreduce(tensor)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_twoshot(num_elements, world_size, num_warmup=5, num_iters=50):
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)
    for _ in range(num_warmup):
        reset_rand(tensor)
        twoshot_collective_allreduce(tensor)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        twoshot_collective_allreduce(tensor)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_naive_twoshot(num_elements, world_size, num_warmup=5, num_iters=50):
    from compressed_twoshot_allreduce import naive_twoshot_allreduce
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)
    for _ in range(num_warmup):
        reset_rand(tensor)
        naive_twoshot_allreduce(tensor)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        naive_twoshot_allreduce(tensor)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_p2p(num_elements, world_size, num_warmup=5, num_iters=50):
    from p2p_twoshot_allreduce import P2PTwoShotAllReduce
    num_elements = (num_elements // world_size) * world_size
    rank = dist.get_rank()
    p2p = P2PTwoShotAllReduce(num_elements, DTYPE, torch.device("cuda"), rank, world_size)
    buf = p2p.buffer[:num_elements]

    for _ in range(num_warmup):
        buf.uniform_(-1.0, 1.0)
        p2p(buf)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        buf.uniform_(-1.0, 1.0)
        p2p(buf)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    p2p.cleanup()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_p2p_nopa(num_elements, world_size, num_warmup=5, num_iters=50):
    """P2P allreduce with cudaDeviceDisablePeerAccess to measure impact."""
    from p2p_twoshot_allreduce import P2PTwoShotAllReduce, _ext
    num_elements = (num_elements // world_size) * world_size
    rank = dist.get_rank()
    p2p = P2PTwoShotAllReduce(num_elements, DTYPE, torch.device("cuda"), rank, world_size)

    # Disable peer access
    _ext.disable_peer_access()

    buf = p2p.buffer[:num_elements]

    for _ in range(num_warmup):
        buf.uniform_(-1.0, 1.0)
        p2p(buf)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        buf.uniform_(-1.0, 1.0)
        p2p(buf)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Re-enable peer access before cleanup
    _ext.enable_peer_access()
    p2p.cleanup()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_comp_p2p(num_elements, world_size, num_warmup=5, num_iters=50):
    from compressed_p2p_allreduce import CompressedP2PAllReduce
    num_elements = (num_elements // world_size) * world_size
    rank = dist.get_rank()
    ar = CompressedP2PAllReduce(num_elements, DTYPE, torch.device("cuda"), rank, world_size)
    buf = ar.buffer[:num_elements]

    for _ in range(num_warmup):
        buf.uniform_(-1.0, 1.0)
        ar(buf)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        buf.uniform_(-1.0, 1.0)
        ar(buf)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ar.cleanup()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_ringp2p(num_elements, world_size, num_warmup=5, num_iters=50):
    from ring_p2p_allreduce import RingP2PAllReduce
    num_elements = (num_elements // world_size) * world_size
    rank = dist.get_rank()
    rp2p = RingP2PAllReduce(num_elements, DTYPE, torch.device("cuda"), rank, world_size,
                            num_pipes=32)
    buf = rp2p.buffer[:num_elements]

    for _ in range(num_warmup):
        buf.uniform_(-1.0, 1.0)
        rp2p(buf)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        buf.uniform_(-1.0, 1.0)
        rp2p(buf)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    rp2p.cleanup()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_compressed(num_elements, world_size, compress_rs, compress_ag,
                     num_warmup=3, num_iters=20):
    from compressed_sendrecv_allreduce import compressed_sendrecv_allreduce, _instances
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)

    for _ in range(num_warmup):
        reset_rand(tensor)
        compressed_sendrecv_allreduce(tensor, compress_rs=compress_rs, compress_ag=compress_ag)
    torch.cuda.synchronize()

    # Timed loop — no stats collection interference
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        compressed_sendrecv_allreduce(tensor, compress_rs=compress_rs, compress_ag=compress_ag)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    cuda_ms = s.elapsed_time(e) / num_iters
    wall_ms = (t1 - t0) / num_iters * 1000

    # Measure compression ratio separately (outside timing)
    key = (compress_rs, compress_ag)
    inst = _instances.get(key)
    comp_ratio = 0.0
    if inst is not None:
        inst.reset_stats()
        for _ in range(3):
            reset_rand(tensor)
            inst(tensor)
        torch.cuda.synchronize()
        comp_ratio = inst.get_comp_ratio()

    return cuda_ms, wall_ms, comp_ratio


def bench_compressed_nosize(num_elements, world_size, compress_rs, compress_ag,
                            num_warmup=3, num_iters=20):
    from compressed_twoshot_allreduce import compressed_nosize_allreduce, _nosize_instances
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)

    for _ in range(num_warmup):
        reset_rand(tensor)
        compressed_nosize_allreduce(tensor, compress_rs=compress_rs, compress_ag=compress_ag)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        compressed_nosize_allreduce(tensor, compress_rs=compress_rs, compress_ag=compress_ag)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    cuda_ms = s.elapsed_time(e) / num_iters
    wall_ms = (t1 - t0) / num_iters * 1000

    # Measure compression ratio separately
    key = (compress_rs, compress_ag)
    inst = _nosize_instances.get(key)
    comp_ratio = 0.0
    if inst is not None:
        inst.reset_stats()
        for _ in range(3):
            reset_rand(tensor)
            inst(tensor)
        torch.cuda.synchronize()
        comp_ratio = inst.get_comp_ratio()

    return cuda_ms, wall_ms, comp_ratio


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["naive", "naive_twoshot", "ring", "rhd", "twoshot", "p2p", "p2p_nopa", "comp_p2p", "ringp2p", "compressed", "compressed_nosize", "all"],
                        default="naive",
                        help="Which send/recv variant to compare against NCCL")
    parser.add_argument("--compress",
                        choices=["both", "rs", "ag"],
                        default="both",
                        help="Where to apply compression (only used with --mode compressed/all): "
                             "rs=ReduceScatter only, ag=AllGather only, both=both (default)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to write results as CSV. "
                             "Defaults to benchmark_<mode>.csv in the current directory.")
    args = parser.parse_args()

    rank, world_size, _ = setup()

    # Parse compression placement
    compress_rs = args.compress in ("both", "rs")
    compress_ag = args.compress in ("both", "ag")

    sizes = [
        # (1 << 10, "1 KB"),
        # (1 << 12, "4 KB"),
        # (1 << 14, "16 KB"),
        # (1 << 16, "64 KB"),
        # (1 << 18, "256 KB"),
        # (1 << 20, "1 MB"),
        (1 << 22, "4 MB"),
        (1 << 23, "8 MB"),
        (1 << 24, "16 MB"),
        (1 << 25, "32 MB"),
        (1 << 26, "64 MB"),
        (1 << 28, "256 MB"),
        (1 << 30, "1 GB"),
    ]

    run_naive = args.mode in ("naive", "all")
    run_naive_twoshot = args.mode in ("naive_twoshot", "all")
    run_ring = args.mode in ("ring", "all")
    run_rhd = args.mode in ("rhd", "all")
    run_twoshot = args.mode in ("twoshot", "all")
    run_p2p = args.mode in ("p2p", "all")
    run_p2p_nopa = args.mode in ("p2p_nopa", "all")
    run_comp_p2p = args.mode in ("comp_p2p", "all")
    run_ringp2p = args.mode in ("ringp2p", "all")
    run_comp = args.mode in ("compressed", "all")
    run_comp_nosize = args.mode in ("compressed_nosize", "all")

    # Build compressed label
    comp_label_map = {"both": "RS+AG", "rs": "RS only", "ag": "AG only"}
    comp_label = comp_label_map[args.compress]

    if rank == 0:
        mode_desc = {"naive": "Naive send/recv",
                     "naive_twoshot": "Naive two-shot (raw all-to-all RS+AG, no compression)",
                     "ring": "Ring send/recv",
                     "rhd": "Recursive halving-doubling",
                     "twoshot": "Two-shot (reduce_scatter + all_gather)",
                     "p2p": "CUDA IPC P2P flat (no NCCL)",
                     "p2p_nopa": "CUDA IPC P2P flat, peer access disabled",
                     "comp_p2p": "Compressed CUDA IPC P2P (dietgpu + IPC)",
                     "ringp2p": "CUDA IPC P2P ring (no NCCL)",
                     "compressed": f"Compressed send/recv (dietgpu, {comp_label})",
                     "compressed_nosize": f"Compressed no-size-exchange (dietgpu, {comp_label})",
                     "all": f"All variants ({comp_label})"}
        print(f"Benchmarking: NCCL AllReduce vs {mode_desc[args.mode]}")
        print(f"World size: {world_size}, dtype: bfloat16, data: uniform[-1, 1]")
        if run_comp or run_comp_nosize:
            print(f"Compression: RS={'ON' if compress_rs else 'OFF'}, "
                  f"AG={'ON' if compress_ag else 'OFF'}")

    # Short column header for compressed
    comp_col = {"both": "Comp", "rs": "Comp-RS", "ag": "Comp-AG"}[args.compress]

    # Collect results
    results = []
    factor = 2.0 * (world_size - 1) / world_size

    for size_bytes, label in sizes:
        num_elements = size_bytes // 2  # bfloat16 = 2 bytes per element

        cuda_nccl, wall_nccl = bench_standard(num_elements)

        cuda_naive = wall_naive = 0.0
        if run_naive:
            cuda_naive, wall_naive = bench_naive(num_elements, world_size)

        cuda_n2s = wall_n2s = 0.0
        if run_naive_twoshot:
            cuda_n2s, wall_n2s = bench_naive_twoshot(num_elements, world_size)

        cuda_ring = wall_ring = 0.0
        if run_ring:
            cuda_ring, wall_ring = bench_ring(num_elements, world_size)

        cuda_rhd = wall_rhd = 0.0
        if run_rhd:
            cuda_rhd, wall_rhd = bench_rhd(num_elements, world_size)

        cuda_twoshot = wall_twoshot = 0.0
        if run_twoshot:
            cuda_twoshot, wall_twoshot = bench_twoshot(num_elements, world_size)

        cuda_p2p = wall_p2p = 0.0
        if run_p2p:
            cuda_p2p, wall_p2p = bench_p2p(num_elements, world_size)

        cuda_p2p_nopa = wall_p2p_nopa = 0.0
        if run_p2p_nopa:
            cuda_p2p_nopa, wall_p2p_nopa = bench_p2p_nopa(num_elements, world_size)

        cuda_comp_p2p = wall_comp_p2p = 0.0
        if run_comp_p2p:
            cuda_comp_p2p, wall_comp_p2p = bench_comp_p2p(num_elements, world_size)

        cuda_ringp2p = wall_ringp2p = 0.0
        if run_ringp2p:
            cuda_ringp2p, wall_ringp2p = bench_ringp2p(num_elements, world_size)

        cuda_comp = wall_comp = comp_ratio = 0.0
        if run_comp:
            cuda_comp, wall_comp, comp_ratio = bench_compressed(
                num_elements, world_size, compress_rs, compress_ag)

        cuda_cns = wall_cns = cns_ratio = 0.0
        if run_comp_nosize:
            cuda_cns, wall_cns, cns_ratio = bench_compressed_nosize(
                num_elements, world_size, compress_rs, compress_ag)

        results.append((label, size_bytes,
                        cuda_nccl, wall_nccl,
                        cuda_naive, wall_naive,
                        cuda_n2s, wall_n2s,
                        cuda_ring, wall_ring,
                        cuda_rhd, wall_rhd,
                        cuda_twoshot, wall_twoshot,
                        cuda_p2p, wall_p2p,
                        cuda_p2p_nopa, wall_p2p_nopa,
                        cuda_comp_p2p, wall_comp_p2p,
                        cuda_ringp2p, wall_ringp2p,
                        cuda_comp, wall_comp, comp_ratio,
                        cuda_cns, wall_cns, cns_ratio))

    if rank != 0:
        dist.destroy_process_group()
        return

    # ── Helper to compute bandwidths ──
    def compute_bw(size_bytes, latency_ms):
        if latency_ms <= 0:
            return 0.0, 0.0
        algbw = size_bytes / (latency_ms / 1000) / 1e9
        busbw = algbw * factor
        return algbw, busbw

    # ── Helper to build columns ──
    cns_col = "NoSize"

    # Simple variants (no ratio column): (flag, label)
    simple_variants = []
    if run_naive:
        simple_variants.append(("naive", "Naive"))
    if run_naive_twoshot:
        simple_variants.append(("n2s", "N2Shot"))
    if run_ring:
        simple_variants.append(("ring", "Ring"))
    if run_rhd:
        simple_variants.append(("rhd", "RHD"))
    if run_twoshot:
        simple_variants.append(("twoshot", "2Shot"))
    if run_p2p:
        simple_variants.append(("p2p", "P2P"))
    if run_p2p_nopa:
        simple_variants.append(("p2p_nopa", "P2PnoPA"))
    if run_comp_p2p:
        simple_variants.append(("comp_p2p", "CompP2P"))
    if run_ringp2p:
        simple_variants.append(("ringp2p", "RingP2P"))
    # Compressed variants (with ratio column): (flag, label)
    comp_variants = []
    if run_comp:
        comp_variants.append(("comp", comp_col))
    if run_comp_nosize:
        comp_variants.append(("cns", cns_col))

    def build_header():
        cols = [f"{'Size':>10s}", f"{'NCCL':>10s}"]
        for _, lbl in simple_variants:
            cols += [f"{lbl:>10s}", f"{'vs NCCL':>8s}"]
        for _, lbl in comp_variants:
            cols += [f"{lbl:>10s}", f"{'vs NCCL':>8s}", f"{'Ratio':>7s}"]
        cols.append(f"{'NCCL AlgBW':>12s}")
        for _, lbl in simple_variants:
            cols.append(f"{lbl + ' AlgBW':>12s}")
        for _, lbl in comp_variants:
            cols.append(f"{lbl + ' AlgBW':>12s}")
        cols.append(f"{'NCCL BusBW':>12s}")
        for _, lbl in simple_variants:
            cols.append(f"{lbl + ' BusBW':>12s}")
        for _, lbl in comp_variants:
            cols.append(f"{lbl + ' BusBW':>12s}")
        return cols

    def unpack_row(row):
        (label, size_bytes,
         cuda_nccl, wall_nccl,
         cuda_naive, wall_naive,
         cuda_n2s, wall_n2s,
         cuda_ring, wall_ring,
         cuda_rhd, wall_rhd,
         cuda_twoshot, wall_twoshot,
         cuda_p2p, wall_p2p,
         cuda_p2p_nopa, wall_p2p_nopa,
         cuda_comp_p2p, wall_comp_p2p,
         cuda_ringp2p, wall_ringp2p,
         cuda_comp, wall_comp, comp_ratio,
         cuda_cns, wall_cns, cns_ratio) = row
        # Build dicts keyed by variant name for easy lookup
        cuda = {"naive": cuda_naive, "n2s": cuda_n2s,
                "ring": cuda_ring, "rhd": cuda_rhd,
                "twoshot": cuda_twoshot, "p2p": cuda_p2p,
                "p2p_nopa": cuda_p2p_nopa, "comp_p2p": cuda_comp_p2p,
                "ringp2p": cuda_ringp2p,
                "comp": cuda_comp, "cns": cuda_cns}
        wall = {"naive": wall_naive, "n2s": wall_n2s,
                "ring": wall_ring, "rhd": wall_rhd,
                "twoshot": wall_twoshot, "p2p": wall_p2p,
                "p2p_nopa": wall_p2p_nopa, "comp_p2p": wall_comp_p2p,
                "ringp2p": wall_ringp2p,
                "comp": wall_comp, "cns": wall_cns}
        ratio = {"comp": comp_ratio, "cns": cns_ratio}
        return label, size_bytes, cuda_nccl, wall_nccl, cuda, wall, ratio

    def render_table(title, use_wall=False):
        print()
        print("=" * 100)
        print(f"  {title}")
        print("=" * 100)
        cols = build_header()
        if use_wall:
            cols_wall = [f"{'Size':>10s}", f"{'NCCL':>10s}"]
            for _, lbl in simple_variants:
                cols_wall += [f"{lbl:>10s}", f"{'vs NCCL':>8s}"]
            for _, lbl in comp_variants:
                cols_wall += [f"{lbl:>10s}", f"{'vs NCCL':>8s}", f"{'Ratio':>7s}"]
            cols_wall.append(f"{'CPU Ovhd':>10s}")
            cols_wall.append(f"{'NCCL AlgBW':>12s}")
            for _, lbl in simple_variants:
                cols_wall.append(f"{lbl + ' AlgBW':>12s}")
            for _, lbl in comp_variants:
                cols_wall.append(f"{lbl + ' AlgBW':>12s}")
            cols_wall.append(f"{'NCCL BusBW':>12s}")
            for _, lbl in simple_variants:
                cols_wall.append(f"{lbl + ' BusBW':>12s}")
            for _, lbl in comp_variants:
                cols_wall.append(f"{lbl + ' BusBW':>12s}")
            cols = cols_wall
        header = " | ".join(cols)
        print(header)
        print("-" * len(header))

        for row in results:
            label, size_bytes, cuda_nccl, wall_nccl, cuda_d, wall_d, ratio_d = unpack_row(row)
            ref_ms = wall_nccl if use_wall else cuda_nccl
            d = wall_d if use_wall else cuda_d

            parts = [f"{label:>10s}", f"{ref_ms:>8.3f}ms"]
            for key, _ in simple_variants:
                v = d[key]
                pct = ref_ms / v * 100 if v > 0 else 0
                parts += [f"{v:>8.3f}ms", f"{pct:>7.1f}%"]
            for key, _ in comp_variants:
                v = d[key]
                pct = ref_ms / v * 100 if v > 0 else 0
                parts += [f"{v:>8.3f}ms", f"{pct:>7.1f}%", f"{ratio_d[key]:>6.1%}"]

            if use_wall:
                parts.append(f"{wall_nccl - cuda_nccl:>+8.3f}ms")

            algbw_nccl, busbw_nccl = compute_bw(size_bytes, ref_ms)
            parts.append(f"{algbw_nccl:>10.2f} GB")
            for key, _ in simple_variants:
                a, _ = compute_bw(size_bytes, d[key])
                parts.append(f"{a:>10.2f} GB")
            for key, _ in comp_variants:
                a, _ = compute_bw(size_bytes, d[key])
                parts.append(f"{a:>10.2f} GB")

            parts.append(f"{busbw_nccl:>10.2f} GB")
            for key, _ in simple_variants:
                _, b = compute_bw(size_bytes, d[key])
                parts.append(f"{b:>10.2f} GB")
            for key, _ in comp_variants:
                _, b = compute_bw(size_bytes, d[key])
                parts.append(f"{b:>10.2f} GB")

            print(" | ".join(parts))

    render_table("CUDA Event Time (GPU-side latency)", use_wall=False)
    render_table("Wall-Clock Time (includes CPU overhead, kernel launch, Python dispatch)", use_wall=True)

    # ── Write results to CSV ──
    # Format: NCCL on its own column, compared method on another column.
    # Section 1: Wall time. Blank row. Section 2: CUDA time.
    csv_path = args.csv or f"benchmark_{args.mode}.csv"

    variants_to_compare = [(k, lbl) for k, lbl in simple_variants]
    variants_to_compare += [(k, lbl) for k, lbl in comp_variants]

    header = [
        "metric", "size_label", "size_bytes", "world_size", "variant",
        "nccl_ms", "variant_ms", "variant_vs_nccl_pct",
        "nccl_algbw_GBps", "variant_algbw_GBps",
        "nccl_busbw_GBps", "variant_busbw_GBps",
        "comp_ratio",
    ]

    def make_rows(metric_name, use_wall):
        rows_out = []
        for row in results:
            label, size_bytes, cuda_nccl, wall_nccl, cuda_d, wall_d, ratio_d = unpack_row(row)
            nccl_ms = wall_nccl if use_wall else cuda_nccl
            d = wall_d if use_wall else cuda_d
            nccl_alg, nccl_bus = compute_bw(size_bytes, nccl_ms)
            for key, vlabel in variants_to_compare:
                v_ms = d[key]
                # Skip variants that were not run
                if v_ms == 0.0 and (cuda_d[key] == 0.0 and wall_d[key] == 0.0):
                    continue
                v_alg, v_bus = compute_bw(size_bytes, v_ms)
                pct = (nccl_ms / v_ms * 100) if v_ms > 0 else 0.0
                rows_out.append([
                    metric_name, label, size_bytes, world_size, vlabel,
                    f"{nccl_ms:.6f}", f"{v_ms:.6f}", f"{pct:.2f}",
                    f"{nccl_alg:.4f}", f"{v_alg:.4f}",
                    f"{nccl_bus:.4f}", f"{v_bus:.4f}",
                    f"{ratio_d.get(key, 0.0):.6f}" if key in ratio_d else "",
                ])
        return rows_out

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in make_rows("wall_ms", use_wall=True):
            writer.writerow(r)
        writer.writerow([])  # blank separator row
        for r in make_rows("cuda_ms", use_wall=False):
            writer.writerow(r)
    print(f"\nResults written to {csv_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
