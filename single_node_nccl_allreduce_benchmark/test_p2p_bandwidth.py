"""
Simple P2P bandwidth test between GPU 0 and GPU 1.
Tests both with and without cudaDeviceEnablePeerAccess.

Usage: python test_p2p_bandwidth.py
"""

import os
import torch
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="p2p_bw_test",
    sources=[os.path.join(_dir, "p2p_bw_test.cu")],
    verbose=False,
)


def bench_copy(src_dev, dst_dev, size_bytes, num_warmup=10, num_iters=100):
    """Benchmark GPU-to-GPU copy, return (ms, GB/s)."""
    src = torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{src_dev}")
    dst = torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{dst_dev}")

    torch.cuda.set_device(dst_dev)
    for _ in range(num_warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / num_iters
    gb_s = size_bytes / (ms / 1000) / 1e9
    return ms, gb_s


def main():
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"
    src, dst = 0, 1

    print(f"P2P bandwidth test: GPU {src} -> GPU {dst}")
    print(f"Can peer access: {bool(_ext.can_peer(src, dst))}")
    print()

    sizes = [
        (1 << 20, "1 MB"),
        (1 << 24, "16 MB"),
        (1 << 26, "64 MB"),
        (1 << 28, "256 MB"),
        (1 << 30, "1 GB"),
    ]

    print(f"{'Size':>8s}  {'P2P ON (ms)':>12s}  {'BW (GB/s)':>10s}  "
          f"{'P2P OFF (ms)':>13s}  {'BW (GB/s)':>10s}  {'Ratio':>7s}")
    print("-" * 75)

    for nbytes, label in sizes:
        # P2P enabled
        _ext.enable_peer(src, dst)
        _ext.enable_peer(dst, src)
        ms_on, bw_on = bench_copy(src, dst, nbytes)

        # P2P disabled
        _ext.disable_peer(src, dst)
        _ext.disable_peer(dst, src)
        ms_off, bw_off = bench_copy(src, dst, nbytes)

        ratio = bw_off / bw_on if bw_on > 0 else 0
        print(f"{label:>8s}  {ms_on:>10.3f}ms  {bw_on:>8.2f} GB  "
              f"{ms_off:>11.3f}ms  {bw_off:>8.2f} GB  {ratio:>6.1%}")

    # Re-enable
    _ext.enable_peer(src, dst)
    _ext.enable_peer(dst, src)


if __name__ == "__main__":
    main()
