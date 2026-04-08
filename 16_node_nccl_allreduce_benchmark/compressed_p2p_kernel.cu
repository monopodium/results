/*
 * Compressed P2P AllReduce via IPC + dietgpu compression.
 *
 * v2 optimizations:
 *   - Fused multi-source accumulate: reads dst once, accumulates all sources,
 *     writes once → 57% less memory traffic vs sequential accumulations.
 *   - All copies launched before batch decompress for GPU pipelining.
 *   - Fixed-ratio transfer (0.75x) eliminates metadata/size exchange.
 *   - GPU-side atomic flag barriers (same as p2p_twoshot).
 *
 * comp_buf layout per rank: [chunk_0 | chunk_1 | ... | chunk_{ws-1}]
 *   Each slot = comp_stride bytes (aligned max compressed size).
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstring>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    TORCH_CHECK(_e == cudaSuccess, #call " failed: ",                  \
                cudaGetErrorString(_e));                                \
} while (0)

static constexpr int MAX_GPUS = 8;

struct CompP2PState {
    /* data buffer IPC (for uncompressed RS) */
    void*            data_bases[MAX_GPUS] = {};
    void*            data_ptrs[MAX_GPUS]  = {};
    __nv_bfloat16**  d_ptrs               = nullptr;

    /* compressed buffer IPC (for compressed AG) */
    uint8_t* comp_ptrs[MAX_GPUS] = {};
    void*    comp_bases[MAX_GPUS] = {};

    int*     local_flag = nullptr;
    void*    flag_bases[MAX_GPUS] = {};
    int*     flag_ptrs[MAX_GPUS]  = {};
    int**    d_flag_ptrs = nullptr;
    int      phase = 0;

    int      rank = -1, ws = 0;
    int      comp_stride = 0;
    bool     ok = false;
};

static CompP2PState g;

/* ═══════════════ Alloc / IPC ═══════════════ */

torch::Tensor alloc_ipc_buffer(int64_t size_bytes, int dev) {
    c10::cuda::CUDAGuard guard(dev);
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size_bytes));
    CUDA_CHECK(cudaMemset(ptr, 0, size_bytes));
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, dev);
    return torch::from_blob(ptr, {size_bytes}, [](void* p){ cudaFree(p); }, opts);
}

torch::Tensor get_ipc_handle(torch::Tensor t) {
    cudaIpcMemHandle_t h;
    CUDA_CHECK(cudaIpcGetMemHandle(&h, t.data_ptr()));
    auto out = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
    std::memcpy(out.data_ptr(), &h, sizeof(h));
    return out;
}

torch::Tensor alloc_flag_buffer() {
    int* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(int)));
    CUDA_CHECK(cudaMemset(ptr, 0, sizeof(int)));
    g.local_flag = ptr;
    cudaIpcMemHandle_t h;
    CUDA_CHECK(cudaIpcGetMemHandle(&h, ptr));
    auto out = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
    std::memcpy(out.data_ptr(), &h, sizeof(h));
    return out;
}

/* ═══════════════ Init / Cleanup ═══════════════ */

void init_comp_p2p(torch::Tensor all_data_handles,
                   torch::Tensor all_comp_handles,
                   torch::Tensor all_flag_handles,
                   torch::Tensor local_data,
                   torch::Tensor local_comp,
                   int rank, int ws, int comp_stride)
{
    TORCH_CHECK(!g.ok);
    TORCH_CHECK(ws <= MAX_GPUS);
    g.rank = rank;  g.ws = ws;  g.phase = 0;
    g.comp_stride = comp_stride;

    /* Open data buffer IPC handles (for uncompressed RS) */
    for (int r = 0; r < ws; r++) {
        if (r == rank) {
            g.data_bases[r] = nullptr;
            g.data_ptrs[r]  = local_data.data_ptr();
        } else {
            cudaIpcMemHandle_t h;
            std::memcpy(&h, all_data_handles[r].data_ptr(), sizeof(h));
            void* base = nullptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&base, h, cudaIpcMemLazyEnablePeerAccess));
            g.data_bases[r] = base;
            g.data_ptrs[r]  = base;
        }
    }
    CUDA_CHECK(cudaMalloc(&g.d_ptrs, ws * sizeof(__nv_bfloat16*)));
    CUDA_CHECK(cudaMemcpy(g.d_ptrs, g.data_ptrs,
                          ws * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice));

    /* Open comp buffer IPC handles (for compressed AG) */
    for (int r = 0; r < ws; r++) {
        if (r == rank) {
            g.comp_bases[r] = nullptr;
            g.comp_ptrs[r]  = static_cast<uint8_t*>(local_comp.data_ptr());
        } else {
            cudaIpcMemHandle_t h;
            std::memcpy(&h, all_comp_handles[r].data_ptr(), sizeof(h));
            void* base = nullptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&base, h, cudaIpcMemLazyEnablePeerAccess));
            g.comp_bases[r] = base;
            g.comp_ptrs[r]  = static_cast<uint8_t*>(base);
        }
    }

    /* Open flag IPC handles */
    for (int r = 0; r < ws; r++) {
        if (r == rank) {
            g.flag_bases[r] = nullptr;
            g.flag_ptrs[r]  = g.local_flag;
        } else {
            cudaIpcMemHandle_t h;
            std::memcpy(&h, all_flag_handles[r].data_ptr(), sizeof(h));
            void* base = nullptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&base, h, cudaIpcMemLazyEnablePeerAccess));
            g.flag_bases[r] = base;
            g.flag_ptrs[r]  = static_cast<int*>(base);
        }
    }
    CUDA_CHECK(cudaMalloc(&g.d_flag_ptrs, ws * sizeof(int*)));
    CUDA_CHECK(cudaMemcpy(g.d_flag_ptrs, g.flag_ptrs,
                          ws * sizeof(int*), cudaMemcpyHostToDevice));
    g.ok = true;
}

void cleanup_comp_p2p() {
    if (!g.ok) return;
    for (int r = 0; r < g.ws; r++) {
        if (r != g.rank && g.data_bases[r])  cudaIpcCloseMemHandle(g.data_bases[r]);
        if (r != g.rank && g.comp_bases[r])  cudaIpcCloseMemHandle(g.comp_bases[r]);
        if (r != g.rank && g.flag_bases[r])  cudaIpcCloseMemHandle(g.flag_bases[r]);
    }
    if (g.d_ptrs)      cudaFree(g.d_ptrs);
    if (g.d_flag_ptrs) cudaFree(g.d_flag_ptrs);
    if (g.local_flag)  cudaFree(g.local_flag);
    g = CompP2PState{};
}

/* ═══════════════ Kernels ═══════════════ */

__global__ void barrier_kernel(int** flag_ptrs, int rank, int ws, int phase) {
    __threadfence_system();
    atomicExch(flag_ptrs[rank], phase);
    for (int r = 0; r < ws; r++) {
        if (r == rank) continue;
        while (atomicAdd(flag_ptrs[r], 0) < phase) {}
    }
    __threadfence_system();
}

void gpu_barrier() {
    TORCH_CHECK(g.ok);
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    barrier_kernel<<<1, 1, 0, s>>>(g.d_flag_ptrs, g.rank, g.ws, ++g.phase);
}

/* ── Sequential RS kernel: read ONE remote GPU, accumulate via __hadd2 ──
 *
 * Native bf16x2 paired addition (1 instruction per 2 elements, SM80+).
 * __ldg() for remote reads.  Staggered launch order avoids NVLink
 * contention -- optimal for large chunks where bandwidth dominates. */
__global__ void __launch_bounds__(256)
rs_one_source(__nv_bfloat16* __restrict__ data,
              __nv_bfloat16** __restrict__ all_ptrs,
              int src, int off, int elems)
{
    constexpr int V = 16;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= elems) return;
    int idx = off + base;

    /* Load remote (__ldg -> read-only / texture cache path) */
    int4 r0 = __ldg(reinterpret_cast<const int4*>(all_ptrs[src] + idx));
    int4 r1 = __ldg(reinterpret_cast<const int4*>(all_ptrs[src] + idx + 8));
    /* Load local */
    int4 l0 = *reinterpret_cast<const int4*>(data + idx);
    int4 l1 = *reinterpret_cast<const int4*>(data + idx + 8);

    /* Native bf16x2 paired add: 8 __hadd2 for 16 elements (was 24 scalar ops) */
    int4 o0, o1;
    __nv_bfloat162* o2a = reinterpret_cast<__nv_bfloat162*>(&o0);
    __nv_bfloat162* r2a = reinterpret_cast<__nv_bfloat162*>(&r0);
    __nv_bfloat162* l2a = reinterpret_cast<__nv_bfloat162*>(&l0);
    #pragma unroll
    for (int i = 0; i < 4; i++) o2a[i] = __hadd2(r2a[i], l2a[i]);

    __nv_bfloat162* o2b = reinterpret_cast<__nv_bfloat162*>(&o1);
    __nv_bfloat162* r2b = reinterpret_cast<__nv_bfloat162*>(&r1);
    __nv_bfloat162* l2b = reinterpret_cast<__nv_bfloat162*>(&l1);
    #pragma unroll
    for (int i = 0; i < 4; i++) o2b[i] = __hadd2(r2b[i], l2b[i]);

    *reinterpret_cast<int4*>(data + idx)     = o0;
    *reinterpret_cast<int4*>(data + idx + 8) = o1;
}

/* ── Fused RS kernel: read ALL remote GPUs at once, accumulate in one pass ──
 *
 * Reads local data once, reads all (ws-1) remote chunks, accumulates in
 * float32 registers for precision, writes back once.
 *
 * Memory traffic vs sequential rs_one_source:
 *   Sequential (ws-1 launches): (ws-1) × (read local + read remote + write local)
 *                                = (ws-1)×3 chunk-sized memory ops  (21 for ws=8)
 *   Fused     (1 launch):       1 × read local + (ws-1) × read remote + 1 × write
 *                                = (ws-1)+2 chunk-sized memory ops    ( 9 for ws=8)
 *   → 57% less HBM traffic + eliminates (ws-2) kernel launches.
 *
 * Uses __ldg() for remote reads (texture cache / read-only path).
 * Float32 accumulators avoid bf16 rounding over 8-way sum.
 */
__global__ void __launch_bounds__(256)
rs_fused(__nv_bfloat16* __restrict__ data,
         __nv_bfloat16** __restrict__ all_ptrs,
         int rank, int ws, int off, int elems)
{
    constexpr int V = 16;               /* 16 bf16 = 32 bytes per thread */
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= elems) return;
    int idx = off + base;

    /* ── Load local data into float accumulators ─�� */
    int4 l0 = *reinterpret_cast<const int4*>(data + idx);
    int4 l1 = *reinterpret_cast<const int4*>(data + idx + 8);
    const __nv_bfloat16* lp0 = reinterpret_cast<const __nv_bfloat16*>(&l0);
    const __nv_bfloat16* lp1 = reinterpret_cast<const __nv_bfloat16*>(&l1);

    float a[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) a[i]   = __bfloat162float(lp0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) a[8+i] = __bfloat162float(lp1[i]);

    /* ── Accumulate from ALL remote GPUs ──
     * Fully-unrolled over MAX_GPUS; branches are warp-uniform (no divergence).
     * __ldg() hints read-only / texture-cache path for NVLink reads. */
    #pragma unroll
    for (int s = 0; s < MAX_GPUS; s++) {
        if (s >= ws || s == rank) continue;
        int4 r0 = __ldg(reinterpret_cast<const int4*>(all_ptrs[s] + idx));
        int4 r1 = __ldg(reinterpret_cast<const int4*>(all_ptrs[s] + idx + 8));
        const __nv_bfloat16* rp0 = reinterpret_cast<const __nv_bfloat16*>(&r0);
        const __nv_bfloat16* rp1 = reinterpret_cast<const __nv_bfloat16*>(&r1);
        #pragma unroll
        for (int i = 0; i < 8; i++) a[i]   += __bfloat162float(rp0[i]);
        #pragma unroll
        for (int i = 0; i < 8; i++) a[8+i] += __bfloat162float(rp1[i]);
    }

    /* ── Convert float32 accumulators → bf16 and store ── */
    int4 o0, o1;
    __nv_bfloat16* op0 = reinterpret_cast<__nv_bfloat16*>(&o0);
    __nv_bfloat16* op1 = reinterpret_cast<__nv_bfloat16*>(&o1);
    #pragma unroll
    for (int i = 0; i < 8; i++) op0[i] = __float2bfloat16(a[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) op1[i] = __float2bfloat16(a[8+i]);

    *reinterpret_cast<int4*>(data + idx)     = o0;
    *reinterpret_cast<int4*>(data + idx + 8) = o1;
}

/* ── Adaptive RS launch ──
 * Small chunks (< 1M elems = 2MB): fused kernel saves 6 launches.
 * Large chunks:  sequential + staggered avoids NVLink contention. */
void run_rs(torch::Tensor t, int64_t n) {
    TORCH_CHECK(g.ok);
    int ws = g.ws, rank = g.rank;
    int chunk = n / ws;
    int my_off = rank * chunk;
    auto* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr());

    constexpr int B = 256, V = 16;
    int grid = (chunk / V + B - 1) / B;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();

    if (chunk < (1 << 20)) {
        rs_fused<<<grid, B, 0, st>>>(p, g.d_ptrs, rank, ws, my_off, chunk);
    } else {
        for (int i = 0; i < ws - 1; i++) {
            int s = (rank + 1 + i) % ws;
            rs_one_source<<<grid, B, 0, st>>>(p, g.d_ptrs, s, my_off, chunk);
        }
    }
    barrier_kernel<<<1, 1, 0, st>>>(g.d_flag_ptrs, rank, ws, ++g.phase);
}

/* Copy fixed-size compressed data from remote rank's comp_buf. */
__global__ void __launch_bounds__(256)
copy_fixed_kernel(uint8_t* __restrict__ dst,
                  const uint8_t* __restrict__ src,
                  int nbytes)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * 16;
    if (base >= nbytes) return;
    if (base + 16 <= nbytes) {
        reinterpret_cast<int4*>(dst + base)[0] =
            reinterpret_cast<const int4*>(src + base)[0];
    } else {
        for (int i = base; i < nbytes; i++) dst[i] = src[i];
    }
}

void copy_from_remote(torch::Tensor dst, int src_rank,
                      int64_t byte_offset, int64_t nbytes) {
    TORCH_CHECK(g.ok);
    const uint8_t* src = g.comp_ptrs[src_rank] + byte_offset;
    int grid = ((int)nbytes + 16 * 256 - 1) / (16 * 256);
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    copy_fixed_kernel<<<grid, 256, 0, s>>>(
        static_cast<uint8_t*>(dst.data_ptr()), src, (int)nbytes);
}

/* ── Fused multi-source accumulate ──
 *
 * dst += src_base[0*stride : 0*stride+n]
 *      + src_base[1*stride : 1*stride+n]
 *      + ...
 *      + src_base[(nsrcs-1)*stride : (nsrcs-1)*stride+n]
 *
 * Reads dst once, accumulates all sources in registers, writes dst once.
 * For 7 sources: 9 memory passes (1 dst + 7 src reads + 1 dst write)
 * vs sequential: 21 passes (7 × {read dst + read src + write dst}).
 */
__global__ void __launch_bounds__(256)
fused_accum_kernel(__nv_bfloat16* __restrict__ dst,
                   const __nv_bfloat16* __restrict__ src_base,
                   int64_t stride,       /* elements between sources */
                   int     nsrcs,
                   int     n)
{
    constexpr int V = 16;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= n) return;

    /* Load dst into float accumulators */
    int4 d0 = *reinterpret_cast<const int4*>(dst + base);
    int4 d1 = *reinterpret_cast<const int4*>(dst + base + 8);
    const __nv_bfloat16* dp0 = reinterpret_cast<const __nv_bfloat16*>(&d0);
    const __nv_bfloat16* dp1 = reinterpret_cast<const __nv_bfloat16*>(&d1);

    float a[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) a[i]   = __bfloat162float(dp0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) a[8+i] = __bfloat162float(dp1[i]);

    /* Accumulate each source */
    for (int s = 0; s < nsrcs; s++) {
        const __nv_bfloat16* src = src_base + s * stride;
        int4 s0 = *reinterpret_cast<const int4*>(src + base);
        int4 s1 = *reinterpret_cast<const int4*>(src + base + 8);
        const __nv_bfloat16* sp0 = reinterpret_cast<const __nv_bfloat16*>(&s0);
        const __nv_bfloat16* sp1 = reinterpret_cast<const __nv_bfloat16*>(&s1);
        #pragma unroll
        for (int i = 0; i < 8; i++) a[i]   += __bfloat162float(sp0[i]);
        #pragma unroll
        for (int i = 0; i < 8; i++) a[8+i] += __bfloat162float(sp1[i]);
    }

    /* Convert back and store */
    int4 o0, o1;
    __nv_bfloat16* op0 = reinterpret_cast<__nv_bfloat16*>(&o0);
    __nv_bfloat16* op1 = reinterpret_cast<__nv_bfloat16*>(&o1);
    #pragma unroll
    for (int i = 0; i < 8; i++) op0[i] = __float2bfloat16(a[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) op1[i] = __float2bfloat16(a[8+i]);

    *reinterpret_cast<int4*>(dst + base)     = o0;
    *reinterpret_cast<int4*>(dst + base + 8) = o1;
}

void fused_accumulate(torch::Tensor dst, torch::Tensor src_base,
                      int64_t stride, int nsrcs, int64_t n) {
    constexpr int B = 256, V = 16;
    int grid = ((int)n / V + B - 1) / B;
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    fused_accum_kernel<<<grid, B, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(src_base.data_ptr()),
        stride, nsrcs, (int)n);
}

/* ═══════════════ pybind ═══════════════ */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc_ipc_buffer",    &alloc_ipc_buffer);
    m.def("get_ipc_handle",      &get_ipc_handle);
    m.def("alloc_flag_buffer",   &alloc_flag_buffer);
    m.def("init_comp_p2p",       &init_comp_p2p);
    m.def("cleanup_comp_p2p",    &cleanup_comp_p2p);
    m.def("gpu_barrier",         &gpu_barrier);
    m.def("run_rs",              &run_rs);
    m.def("copy_from_remote",    &copy_from_remote);
    m.def("fused_accumulate",    &fused_accumulate);
}
