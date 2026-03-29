"""
Benchmark: NCCL AllReduce vs send/recv AllReduce variants.
Reports both CUDA event time (GPU-side) and wall-clock time (includes CPU overhead).

Usage:
  torchrun ... benchmark_both.py --mode naive                    # NCCL vs Naive
  torchrun ... benchmark_both.py --mode optimized                # NCCL vs Optimized
  torchrun ... benchmark_both.py --mode compressed               # NCCL vs Compressed (both RS+AG)
  torchrun ... benchmark_both.py --mode compressed --compress rs # Compress ReduceScatter only
  torchrun ... benchmark_both.py --mode compressed --compress ag # Compress AllGather only
  torchrun ... benchmark_both.py --mode compressed_nosize        # No size exchange (fixed 0.65 ratio)
  torchrun ... benchmark_both.py --mode all                      # All variants
"""

import argparse
import os
import time
import torch
import torch.distributed as dist

from naive_sendrecv_allreduce import naive_sendrecv_allreduce
from twoshot_allreduce import TwoShotAllReduce
from compressed_sendrecv_allreduce import compressed_sendrecv_allreduce
from compressed_nosize_allreduce import compressed_nosize_allreduce


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
    return torch.empty(num_elements, dtype=DTYPE, device="cuda").uniform_(-1.0, 1.0)

def reset_rand(tensor):
    """Re-fill tensor with fresh random data in [-1, 1]."""
    tensor.uniform_(-1.0, 1.0)


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


_twoshot_instance = None

def bench_optimized(num_elements, world_size, num_warmup=5, num_iters=50):
    global _twoshot_instance
    num_elements = (num_elements // world_size) * world_size
    tensor = make_rand_tensor(num_elements)

    if _twoshot_instance is None or _twoshot_instance.rs_recv_buf.numel() < num_elements // 2:
        _twoshot_instance = TwoShotAllReduce(
            max(num_elements, 1 << 28), DTYPE, tensor.device, num_pipelines=8
        )

    for _ in range(num_warmup):
        reset_rand(tensor)
        _twoshot_instance(tensor)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        reset_rand(tensor)
        _twoshot_instance(tensor)
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e) / num_iters, (t1 - t0) / num_iters * 1000


def bench_compressed(num_elements, world_size, compress_rs, compress_ag,
                     num_warmup=3, num_iters=20):
    from compressed_sendrecv_allreduce import _instances
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
    from compressed_nosize_allreduce import _nosize_instances
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
                        choices=["naive", "optimized", "compressed", "compressed_nosize", "all"],
                        default="naive",
                        help="Which send/recv variant to compare against NCCL")
    parser.add_argument("--compress",
                        choices=["both", "rs", "ag"],
                        default="both",
                        help="Where to apply compression (only used with --mode compressed/all): "
                             "rs=ReduceScatter only, ag=AllGather only, both=both (default)")
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
    run_opt = args.mode in ("optimized", "all")
    run_comp = args.mode in ("compressed", "all")
    run_comp_nosize = args.mode in ("compressed_nosize", "all")

    # Build compressed label
    comp_label_map = {"both": "RS+AG", "rs": "RS only", "ag": "AG only"}
    comp_label = comp_label_map[args.compress]

    if rank == 0:
        mode_desc = {"naive": "Naive send/recv",
                     "optimized": "Optimized send/recv",
                     "compressed": f"Compressed send/recv (dietgpu, {comp_label})",
                     "compressed_nosize": f"Compressed no-size-exchange (dietgpu, {comp_label})",
                     "all": f"Naive + Optimized + Compressed + CompNoSize ({comp_label})"}
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

        cuda_opt = wall_opt = 0.0
        if run_opt:
            cuda_opt, wall_opt = bench_optimized(num_elements, world_size)

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
                        cuda_opt, wall_opt,
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

    def build_header():
        cols = [f"{'Size':>10s}", f"{'NCCL':>10s}"]
        if run_naive:
            cols += [f"{'Naive':>10s}", f"{'vs NCCL':>8s}"]
        if run_opt:
            cols += [f"{'Opt':>10s}", f"{'vs NCCL':>8s}"]
        if run_comp:
            cols += [f"{comp_col:>10s}", f"{'vs NCCL':>8s}", f"{'Ratio':>7s}"]
        if run_comp_nosize:
            cols += [f"{cns_col:>10s}", f"{'vs NCCL':>8s}", f"{'Ratio':>7s}"]
        # AlgBW columns
        cols.append(f"{'NCCL AlgBW':>12s}")
        if run_naive:
            cols.append(f"{'Naive AlgBW':>12s}")
        if run_opt:
            cols.append(f"{'Opt AlgBW':>12s}")
        if run_comp:
            cols.append(f"{comp_col + ' AlgBW':>12s}")
        if run_comp_nosize:
            cols.append(f"{cns_col + ' AlgBW':>12s}")
        # BusBW columns
        cols.append(f"{'NCCL BusBW':>12s}")
        if run_naive:
            cols.append(f"{'Naive BusBW':>12s}")
        if run_opt:
            cols.append(f"{'Opt BusBW':>12s}")
        if run_comp:
            cols.append(f"{comp_col + ' BusBW':>12s}")
        if run_comp_nosize:
            cols.append(f"{cns_col + ' BusBW':>12s}")
        return cols

    # ── Table 1: CUDA event time ──
    print()
    print("=" * 100)
    print("  CUDA Event Time (GPU-side latency)")
    print("=" * 100)
    cols = build_header()
    header = " | ".join(cols)
    print(header)
    print("-" * len(header))

    for row in results:
        (label, size_bytes,
         cuda_nccl, wall_nccl,
         cuda_naive, wall_naive,
         cuda_opt, wall_opt,
         cuda_comp, wall_comp, comp_ratio,
         cuda_cns, wall_cns, cns_ratio) = row

        algbw_nccl, busbw_nccl = compute_bw(size_bytes, cuda_nccl)
        parts = [f"{label:>10s}", f"{cuda_nccl:>8.3f}ms"]

        if run_naive:
            pct = cuda_nccl / cuda_naive * 100 if cuda_naive > 0 else 0
            parts += [f"{cuda_naive:>8.3f}ms", f"{pct:>7.1f}%"]
        if run_opt:
            pct = cuda_nccl / cuda_opt * 100 if cuda_opt > 0 else 0
            parts += [f"{cuda_opt:>8.3f}ms", f"{pct:>7.1f}%"]
        if run_comp:
            pct = cuda_nccl / cuda_comp * 100 if cuda_comp > 0 else 0
            parts += [f"{cuda_comp:>8.3f}ms", f"{pct:>7.1f}%",
                      f"{comp_ratio:>6.1%}"]
        if run_comp_nosize:
            pct = cuda_nccl / cuda_cns * 100 if cuda_cns > 0 else 0
            parts += [f"{cuda_cns:>8.3f}ms", f"{pct:>7.1f}%",
                      f"{cns_ratio:>6.1%}"]

        # AlgBW columns
        parts.append(f"{algbw_nccl:>10.2f} GB")
        if run_naive:
            a, _ = compute_bw(size_bytes, cuda_naive)
            parts.append(f"{a:>10.2f} GB")
        if run_opt:
            a, _ = compute_bw(size_bytes, cuda_opt)
            parts.append(f"{a:>10.2f} GB")
        if run_comp:
            a, _ = compute_bw(size_bytes, cuda_comp)
            parts.append(f"{a:>10.2f} GB")
        if run_comp_nosize:
            a, _ = compute_bw(size_bytes, cuda_cns)
            parts.append(f"{a:>10.2f} GB")

        # BusBW columns
        parts.append(f"{busbw_nccl:>10.2f} GB")
        if run_naive:
            _, b = compute_bw(size_bytes, cuda_naive)
            parts.append(f"{b:>10.2f} GB")
        if run_opt:
            _, b = compute_bw(size_bytes, cuda_opt)
            parts.append(f"{b:>10.2f} GB")
        if run_comp:
            _, b = compute_bw(size_bytes, cuda_comp)
            parts.append(f"{b:>10.2f} GB")
        if run_comp_nosize:
            _, b = compute_bw(size_bytes, cuda_cns)
            parts.append(f"{b:>10.2f} GB")

        print(" | ".join(parts))

    # ── Table 2: Wall-clock time ──
    print()
    print("=" * 100)
    print("  Wall-Clock Time (includes CPU overhead, kernel launch, Python dispatch)")
    print("=" * 100)
    cols = [f"{'Size':>10s}", f"{'NCCL':>10s}"]
    if run_naive:
        cols += [f"{'Naive':>10s}", f"{'vs NCCL':>8s}"]
    if run_opt:
        cols += [f"{'Opt':>10s}", f"{'vs NCCL':>8s}"]
    if run_comp:
        cols += [f"{comp_col:>10s}", f"{'vs NCCL':>8s}", f"{'Ratio':>7s}"]
    if run_comp_nosize:
        cols += [f"{cns_col:>10s}", f"{'vs NCCL':>8s}", f"{'Ratio':>7s}"]
    cols.append(f"{'CPU Ovhd':>10s}")
    # AlgBW columns
    cols.append(f"{'NCCL AlgBW':>12s}")
    if run_naive:
        cols.append(f"{'Naive AlgBW':>12s}")
    if run_opt:
        cols.append(f"{'Opt AlgBW':>12s}")
    if run_comp:
        cols.append(f"{comp_col + ' AlgBW':>12s}")
    if run_comp_nosize:
        cols.append(f"{cns_col + ' AlgBW':>12s}")
    # BusBW columns
    cols.append(f"{'NCCL BusBW':>12s}")
    if run_naive:
        cols.append(f"{'Naive BusBW':>12s}")
    if run_opt:
        cols.append(f"{'Opt BusBW':>12s}")
    if run_comp:
        cols.append(f"{comp_col + ' BusBW':>12s}")
    if run_comp_nosize:
        cols.append(f"{cns_col + ' BusBW':>12s}")
    header = " | ".join(cols)
    print(header)
    print("-" * len(header))

    for row in results:
        (label, size_bytes,
         cuda_nccl, wall_nccl,
         cuda_naive, wall_naive,
         cuda_opt, wall_opt,
         cuda_comp, wall_comp, comp_ratio,
         cuda_cns, wall_cns, cns_ratio) = row

        overhead = wall_nccl - cuda_nccl
        parts = [f"{label:>10s}", f"{wall_nccl:>8.3f}ms"]

        if run_naive:
            pct = wall_nccl / wall_naive * 100 if wall_naive > 0 else 0
            parts += [f"{wall_naive:>8.3f}ms", f"{pct:>7.1f}%"]
        if run_opt:
            pct = wall_nccl / wall_opt * 100 if wall_opt > 0 else 0
            parts += [f"{wall_opt:>8.3f}ms", f"{pct:>7.1f}%"]
        if run_comp:
            pct = wall_nccl / wall_comp * 100 if wall_comp > 0 else 0
            parts += [f"{wall_comp:>8.3f}ms", f"{pct:>7.1f}%",
                      f"{comp_ratio:>6.1%}"]
        if run_comp_nosize:
            pct = wall_nccl / wall_cns * 100 if wall_cns > 0 else 0
            parts += [f"{wall_cns:>8.3f}ms", f"{pct:>7.1f}%",
                      f"{cns_ratio:>6.1%}"]

        parts.append(f"{overhead:>+8.3f}ms")

        # AlgBW columns (based on wall time)
        algbw_nccl, busbw_nccl = compute_bw(size_bytes, wall_nccl)
        parts.append(f"{algbw_nccl:>10.2f} GB")
        if run_naive:
            a, _ = compute_bw(size_bytes, wall_naive)
            parts.append(f"{a:>10.2f} GB")
        if run_opt:
            a, _ = compute_bw(size_bytes, wall_opt)
            parts.append(f"{a:>10.2f} GB")
        if run_comp:
            a, _ = compute_bw(size_bytes, wall_comp)
            parts.append(f"{a:>10.2f} GB")
        if run_comp_nosize:
            a, _ = compute_bw(size_bytes, wall_cns)
            parts.append(f"{a:>10.2f} GB")

        # BusBW columns (based on wall time)
        parts.append(f"{busbw_nccl:>10.2f} GB")
        if run_naive:
            _, b = compute_bw(size_bytes, wall_naive)
            parts.append(f"{b:>10.2f} GB")
        if run_opt:
            _, b = compute_bw(size_bytes, wall_opt)
            parts.append(f"{b:>10.2f} GB")
        if run_comp:
            _, b = compute_bw(size_bytes, wall_comp)
            parts.append(f"{b:>10.2f} GB")
        if run_comp_nosize:
            _, b = compute_bw(size_bytes, wall_cns)
            parts.append(f"{b:>10.2f} GB")

        print(" | ".join(parts))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
