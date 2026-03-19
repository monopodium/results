#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__              \
                << " code=" << static_cast<int>(err__)                          \
                << " \"" << cudaGetErrorString(err__) << "\"\n";                \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

// Layout:
// fp8 a: [7]=sign, [6:3]=exp, [2:0]=mant
// fp8 b: [7]=sign, [6:3]=exp, [2:0]=mant
//
// packed 16-bit "bf16 bit container":
// [15]    = a.sign
// [14:11] = a.exp
// [10:7]  = b.exp
// [6:4]   = a.mant
// [3]     = b.sign
// [2:0]   = b.mant

__device__ __forceinline__ uint16_t PackTwoFp8E4M3FN(uint8_t a, uint8_t b) {
  const uint16_t a_sign = static_cast<uint16_t>((a >> 7) & 0x1);
  const uint16_t a_exp  = static_cast<uint16_t>((a >> 3) & 0xF);
  const uint16_t a_man  = static_cast<uint16_t>(a & 0x7);

  const uint16_t b_sign = static_cast<uint16_t>((b >> 7) & 0x1);
  const uint16_t b_exp  = static_cast<uint16_t>((b >> 3) & 0xF);
  const uint16_t b_man  = static_cast<uint16_t>(b & 0x7);

  uint16_t out = 0;
  out |= (a_sign << 15);
  out |= (a_exp  << 11);
  out |= (b_exp  << 7);
  out |= (a_man  << 4);
  out |= (b_sign << 3);
  out |=  b_man;
  return out;
}

__device__ __forceinline__ void UnpackTwoFp8E4M3FN(
    uint16_t in, uint8_t* a, uint8_t* b) {
  const uint8_t a_sign = static_cast<uint8_t>((in >> 15) & 0x1);
  const uint8_t a_exp  = static_cast<uint8_t>((in >> 11) & 0xF);
  const uint8_t b_exp  = static_cast<uint8_t>((in >> 7)  & 0xF);
  const uint8_t a_man  = static_cast<uint8_t>((in >> 4)  & 0x7);
  const uint8_t b_sign = static_cast<uint8_t>((in >> 3)  & 0x1);
  const uint8_t b_man  = static_cast<uint8_t>( in        & 0x7);

  *a = static_cast<uint8_t>((a_sign << 7) | (a_exp << 3) | a_man);
  *b = static_cast<uint8_t>((b_sign << 7) | (b_exp << 3) | b_man);
}

__global__ void pack_fp8_e4m3fn_to_bf16_bits(const uint8_t* __restrict__ fp8_in,
                                             uint16_t* __restrict__ bf16_out,
                                             size_t n_bytes) {
  const size_t num_pairs = n_bytes >> 1;
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (; idx < num_pairs; idx += stride) {
    const uint8_t a = fp8_in[2 * idx];
    const uint8_t b = fp8_in[2 * idx + 1];
    bf16_out[idx] = PackTwoFp8E4M3FN(a, b);
  }
}

__global__ void unpack_bf16_bits_to_fp8_e4m3fn(
    const uint16_t* __restrict__ bf16_in,
    uint8_t* __restrict__ fp8_out,
    size_t n_bytes) {
  const size_t num_pairs = n_bytes >> 1;
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (; idx < num_pairs; idx += stride) {
    uint8_t a, b;
    UnpackTwoFp8E4M3FN(bf16_in[idx], &a, &b);
    fp8_out[2 * idx] = a;
    fp8_out[2 * idx + 1] = b;
  }
}

static void FillRandomBytes(std::vector<uint8_t>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<uint8_t>(dist(rng));
  }
}

static void PrintExample(const std::vector<uint8_t>& fp8_in,
                         const std::vector<uint16_t>& packed,
                         size_t count) {
  std::cout << "\nExamples:\n";
  for (size_t i = 0; i < count; ++i) {
    uint8_t a = fp8_in[2 * i];
    uint8_t b = fp8_in[2 * i + 1];
    uint16_t p = packed[i];
    std::cout << "pair " << i
              << "  a=0x" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(a)
              << "  b=0x" << std::setw(2)
              << static_cast<int>(b)
              << "  packed=0x" << std::setw(4)
              << static_cast<int>(p)
              << std::dec << "\n";
  }
}

static float BenchmarkPack(const uint8_t* d_fp8_in,
                           uint16_t* d_bf16_out,
                           size_t n_bytes,
                           int warmup,
                           int iters,
                           int blocks,
                           int threads) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    pack_fp8_e4m3fn_to_bf16_bits<<<blocks, threads>>>(d_fp8_in, d_bf16_out,
                                                      n_bytes);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    pack_fp8_e4m3fn_to_bf16_bits<<<blocks, threads>>>(d_fp8_in, d_bf16_out,
                                                      n_bytes);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / static_cast<float>(iters);
}

static float BenchmarkUnpack(const uint16_t* d_bf16_in,
                             uint8_t* d_fp8_out,
                             size_t n_bytes,
                             int warmup,
                             int iters,
                             int blocks,
                             int threads) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    unpack_bf16_bits_to_fp8_e4m3fn<<<blocks, threads>>>(d_bf16_in, d_fp8_out,
                                                        n_bytes);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    unpack_bf16_bits_to_fp8_e4m3fn<<<blocks, threads>>>(d_bf16_in, d_fp8_out,
                                                        n_bytes);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / static_cast<float>(iters);
}

int main(int argc, char** argv) {
  // Default: 256 MiB fp8 input
  size_t n_bytes = 256ull * 1024ull * 1024ull;
  if (argc >= 2) {
    n_bytes = std::strtoull(argv[1], nullptr, 10);
  }
  if (n_bytes % 2 != 0) {
    std::cerr << "Input length must be even.\n";
    return EXIT_FAILURE;
  }

  const int warmup = 10;
  const int iters = 100;
  const int threads = 256;

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  const size_t num_pairs = n_bytes / 2;
  const int blocks = std::min<int>(
      static_cast<int>((num_pairs + threads - 1) / threads),
      prop.multiProcessorCount * 8);

  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Input fp8 bytes: " << n_bytes << "\n";
  std::cout << "Packed bf16 words: " << num_pairs << "\n";
  std::cout << "Blocks: " << blocks << ", Threads: " << threads << "\n";

  std::vector<uint8_t> h_fp8_in(n_bytes);
  std::vector<uint16_t> h_packed(num_pairs);
  std::vector<uint8_t> h_fp8_out(n_bytes);

  FillRandomBytes(h_fp8_in, 12345);

  uint8_t* d_fp8_in = nullptr;
  uint16_t* d_bf16 = nullptr;
  uint8_t* d_fp8_out = nullptr;

  CHECK_CUDA(cudaMalloc(&d_fp8_in, n_bytes));
  CHECK_CUDA(cudaMalloc(&d_bf16, num_pairs * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc(&d_fp8_out, n_bytes));

  CHECK_CUDA(cudaMemcpy(d_fp8_in, h_fp8_in.data(), n_bytes,
                        cudaMemcpyHostToDevice));

  // Correctness: pack then unpack
  pack_fp8_e4m3fn_to_bf16_bits<<<blocks, threads>>>(d_fp8_in, d_bf16, n_bytes);
  CHECK_CUDA(cudaGetLastError());

  unpack_bf16_bits_to_fp8_e4m3fn<<<blocks, threads>>>(d_bf16, d_fp8_out,
                                                      n_bytes);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_packed.data(), d_bf16, num_pairs * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_fp8_out.data(), d_fp8_out, n_bytes,
                        cudaMemcpyDeviceToHost));

  size_t mismatch = 0;
  for (size_t i = 0; i < n_bytes; ++i) {
    if (h_fp8_in[i] != h_fp8_out[i]) {
      mismatch = i + 1;
      break;
    }
  }

  if (mismatch != 0) {
    std::cerr << "Correctness check FAILED at byte " << (mismatch - 1)
              << ": in=0x" << std::hex << static_cast<int>(h_fp8_in[mismatch - 1])
              << " out=0x" << static_cast<int>(h_fp8_out[mismatch - 1])
              << std::dec << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Correctness check: PASS\n";
  PrintExample(h_fp8_in, h_packed, 8);

  // Benchmark
  const float pack_ms = BenchmarkPack(d_fp8_in, d_bf16, n_bytes, warmup, iters,
                                      blocks, threads);
  const float unpack_ms = BenchmarkUnpack(d_bf16, d_fp8_out, n_bytes, warmup,
                                          iters, blocks, threads);

  // Effective traffic:
  // pack:   read N bytes fp8, write N bytes packed (N/2 words = N bytes) => 2N
  // unpack: read N bytes packed, write N bytes fp8 => 2N
  const double total_bytes_per_kernel = 2.0 * static_cast<double>(n_bytes);

  const double pack_gbps =
      total_bytes_per_kernel / (pack_ms * 1e-3) / 1e9;
  const double unpack_gbps =
      total_bytes_per_kernel / (unpack_ms * 1e-3) / 1e9;

  std::cout << "\nBenchmark results:\n";
  std::cout << "Pack   avg time: " << pack_ms << " ms, effective throughput: "
            << pack_gbps << " GB/s\n";
  std::cout << "Unpack avg time: " << unpack_ms << " ms, effective throughput: "
            << unpack_gbps << " GB/s\n";

  CHECK_CUDA(cudaFree(d_fp8_in));
  CHECK_CUDA(cudaFree(d_bf16));
  CHECK_CUDA(cudaFree(d_fp8_out));

  return 0;
}