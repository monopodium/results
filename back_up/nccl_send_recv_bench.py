import torch
import torch.distributed as dist
import time
import argparse
import csv


def run_single_size(size_mb, warmup, iters):
    rank = dist.get_rank()

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    numel = size_mb * 1024 * 1024 // 4
    tensor = torch.randn(numel, dtype=torch.float32, device=device)

    dist.barrier()

    # warmup
    for _ in range(warmup):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)

    dist.barrier()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)

    torch.cuda.synchronize()
    end = time.time()

    if rank == 0:
        elapsed = end - start

        latency = elapsed / iters
        total_bytes = size_mb * 1024 * 1024 * iters
        throughput = total_bytes / elapsed / 1e9

        return latency, throughput

    return None, None


def benchmark(size_list, warmup, iters, output_file):
    rank = dist.get_rank()

    results = []

    for size in size_list:
        latency, throughput = run_single_size(size, warmup, iters)

        if rank == 0:
            print(f"Size {size} MB | Latency {latency*1000:.3f} ms | BW {throughput:.2f} GB/s")
            results.append((size, latency, throughput))

    if rank == 0:
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["data_size_MB", "latency_sec", "throughput_GBps"])
            writer.writerows(results)

        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 512, 1024],
        help="Message sizes in MB",
    )

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=str, default="nccl_results.csv")

    args = parser.parse_args()

    dist.init_process_group("nccl")

    world = dist.get_world_size()
    assert world == 2, "Benchmark requires exactly 2 ranks"

    benchmark(args.sizes, args.warmup, args.iters, args.output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()