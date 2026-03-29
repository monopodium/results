"""Test dietgpu compress → P2P exchange → decompress across nodes.

Node 0: torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=172.31.12.80 --master_port=29500 test_p2p_dietgpu.py
Node 1: torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=172.31.12.80 --master_port=29500 test_p2p_dietgpu.py
"""
import os, glob, torch, torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
peer = 1 - rank
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load dietgpu
so_files = glob.glob("/home/ubuntu/efs/shuangma/uccl/thirdparty/dietgpu/build/lib.*/p2p_dietgpu*.so")
torch.ops.load_library(so_files[0])

COMP_RATIO = 0.7
temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=device)

print(f"[rank{rank}] Starting tests...", flush=True)

# Test 1: Raw P2P sanity check
print(f"\n=== Test 1: Raw P2P exchange (no compression) ===", flush=True)
send_t = torch.full([1024], float(rank + 1), dtype=torch.bfloat16, device=device)
recv_t = torch.zeros([1024], dtype=torch.bfloat16, device=device)
ops = [dist.P2POp(dist.isend, send_t, peer), dist.P2POp(dist.irecv, recv_t, peer)]
for r in dist.batch_isend_irecv(ops):
    r.wait()
torch.cuda.synchronize()
expected = float(peer + 1)
ok = torch.allclose(recv_t, torch.full_like(recv_t, expected))
print(f"[rank{rank}] Raw P2P: received={recv_t[0].item()}, expected={expected}, OK={ok}", flush=True)

# Test 2: Compress locally, decompress locally (no P2P)
print(f"\n=== Test 2: Local compress/decompress ===", flush=True)
for n in [1 << 22, 1 << 24]:
    half = n // 2
    data = torch.empty(half, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    _, max_comp = torch.ops.dietgpu.max_float_compressed_output_size([data])
    comp_buf = torch.empty([1, max_comp], dtype=torch.uint8, device=device)
    comp_sizes = torch.zeros([1], dtype=torch.int, device=device)
    _, comp_sizes, _ = torch.ops.dietgpu.compress_data(
        True, [data], False, temp_mem, comp_buf, comp_sizes)
    torch.cuda.synchronize()
    actual = comp_sizes[0].item()
    fixed = int(half * 2 * COMP_RATIO)

    # Decompress with padded buffer
    comp_data = comp_buf[0, :fixed].contiguous()
    out = torch.empty(half, dtype=torch.bfloat16, device=device)
    ds = torch.empty([1], dtype=torch.uint8, device=device)
    dz = torch.empty([1], dtype=torch.int32, device=device)
    torch.ops.dietgpu.decompress_data(True, [comp_data], [out], False, temp_mem, ds, dz)
    torch.cuda.synchronize()
    match = torch.equal(data, out)
    print(f"[rank{rank}] n={n}: actual={actual}, fixed={fixed}, ratio={actual/(half*2):.4f}, match={match}", flush=True)

# Test 3: Compress → P2P exchange → decompress (the actual flow)
print(f"\n=== Test 3: Compress → P2P → Decompress ===", flush=True)
for n in [1 << 22, 1 << 24]:
    half = n // 2
    data = torch.empty(half, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    _, max_comp = torch.ops.dietgpu.max_float_compressed_output_size([data])

    comp_buf = torch.empty([1, max_comp], dtype=torch.uint8, device=device)
    comp_recv = torch.empty([max_comp], dtype=torch.uint8, device=device)
    comp_sizes = torch.zeros([1], dtype=torch.int, device=device)

    # Compress
    _, comp_sizes, _ = torch.ops.dietgpu.compress_data(
        True, [data], False, temp_mem, comp_buf, comp_sizes)
    torch.cuda.synchronize()
    actual = comp_sizes[0].item()
    fixed = int(half * 2 * COMP_RATIO)
    print(f"[rank{rank}] n={n}: compressed={actual}, fixed={fixed}", flush=True)

    # P2P exchange with fixed_bytes
    comp_send = comp_buf[0, :fixed].contiguous()
    comp_recv_slice = comp_recv[:fixed]
    ops = [
        dist.P2POp(dist.isend, comp_send, peer),
        dist.P2POp(dist.irecv, comp_recv_slice, peer),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    torch.cuda.synchronize()
    print(f"[rank{rank}] P2P done", flush=True)

    # Decompress received data
    out = torch.empty(half, dtype=torch.bfloat16, device=device)
    ds = torch.empty([1], dtype=torch.uint8, device=device)
    dz = torch.empty([1], dtype=torch.int32, device=device)
    try:
        torch.ops.dietgpu.decompress_data(
            True, [comp_recv_slice], [out], False, temp_mem, ds, dz)
        torch.cuda.synchronize()
        print(f"[rank{rank}] Decompress OK, n_out={dz[0].item()}", flush=True)
    except RuntimeError as e:
        print(f"[rank{rank}] Decompress FAILED: {e}", flush=True)

        # Diagnostic: check first 64 bytes of received vs sent
        torch.cuda.synchronize()
        print(f"[rank{rank}] comp_send[:32] = {comp_send[:32].tolist()}", flush=True)
        print(f"[rank{rank}] comp_recv[:32] = {comp_recv_slice[:32].tolist()}", flush=True)

dist.barrier()
if rank == 0:
    print("\nAll tests done!", flush=True)
dist.destroy_process_group()
