"""Quick test: does dietgpu compress/decompress work at all on this machine?"""
import os, glob, torch

# Load the .so
so_dir = "/home/ubuntu/efs/shuangma/uccl/thirdparty/dietgpu/build"
so_files = glob.glob(os.path.join(so_dir, "lib.*", "p2p_dietgpu*.so"))
if not so_files:
    raise RuntimeError(f"No .so found in {so_dir}")
print(f"Loading: {so_files[0]}")
torch.ops.load_library(so_files[0])

device = torch.device("cuda:0")

for n in [1024, 1 << 20, 1 << 22]:
    print(f"\n--- Testing n={n} elements ---")
    data = torch.randn(n, dtype=torch.bfloat16, device=device)

    # Get max compressed size
    _, max_comp = torch.ops.dietgpu.max_float_compressed_output_size([data])
    print(f"  max_comp_cols = {max_comp}")

    # Allocate buffers
    comp_buf = torch.empty([1, max_comp], dtype=torch.uint8, device=device)
    comp_sizes = torch.zeros([1], dtype=torch.int, device=device)
    temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=device)

    # Compress
    _, comp_sizes, _ = torch.ops.dietgpu.compress_data(
        True, [data], False, temp_mem, comp_buf, comp_sizes
    )
    torch.cuda.synchronize()
    actual_size = comp_sizes[0].item()
    ratio = actual_size / (n * 2)
    print(f"  compressed: {actual_size} bytes, ratio: {ratio:.3f} ({ratio*100:.1f}%)")

    # Decompress
    comp_data = comp_buf[0, :actual_size].contiguous()
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    decomp_status = torch.empty([1], dtype=torch.uint8, device=device)
    decomp_sizes = torch.empty([1], dtype=torch.int32, device=device)

    torch.ops.dietgpu.decompress_data(
        True, [comp_data], [out], False, temp_mem, decomp_status, decomp_sizes
    )
    torch.cuda.synchronize()
    print(f"  decomp_sizes: {decomp_sizes[0].item()}")

    # Verify
    match = torch.equal(data, out)
    max_diff = (data.float() - out.float()).abs().max().item()
    print(f"  exact match: {match}, max_diff: {max_diff}")

    # Also test: decompress with OVER-ESTIMATED size (like nosize allreduce does)
    fixed_bytes = int(n * 2 * 0.7)
    if fixed_bytes >= actual_size:
        comp_data_padded = comp_buf[0, :fixed_bytes].contiguous()
        out2 = torch.empty(n, dtype=torch.bfloat16, device=device)
        torch.ops.dietgpu.decompress_data(
            True, [comp_data_padded], [out2], False, temp_mem, decomp_status, decomp_sizes
        )
        torch.cuda.synchronize()
        match2 = torch.equal(data, out2)
        print(f"  padded decompress (fixed_bytes={fixed_bytes}): match={match2}")
    else:
        print(f"  WARNING: actual_size ({actual_size}) > fixed_bytes ({fixed_bytes})! 0.7 ratio NOT enough!")

print("\nAll tests passed!")
