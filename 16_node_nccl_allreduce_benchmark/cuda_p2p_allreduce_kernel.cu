/*
 * Custom CUDA two-shot AllReduce via IPC peer-to-peer memory access.
 *
 * v4: Sequential-source RS & AG — read from ONE remote GPU at a time,
 *     eliminating NVLink contention. Each link runs at full per-pair bandwidth.
 *     GPU-side atomic flag barriers.
 *
 *     RS: 7 kernels (one per remote GPU), accumulate locally
 *     Barrier (GPU flags)
 *     AG: 7 kernels (one per remote GPU), copy
 *     Barrier
 *     Total: 16 kernel launches for 8 GPUs.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstring>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    TORCH_CHECK(_e == cudaSuccess, #call " failed: ",                  \
                cudaGetErrorString(_e));                                \
} while (0)

static constexpr int MAX_GPUS = 8;

struct P2PState {
    void*            bases[MAX_GPUS] = {};
    void*            ptrs[MAX_GPUS]  = {};
    __nv_bfloat16**  d_ptrs          = nullptr;

    int*             local_flag      = nullptr;
    void*            flag_bases[MAX_GPUS] = {};
    int*             flag_ptrs[MAX_GPUS]  = {};
    int**            d_flag_ptrs     = nullptr;
    int              phase           = 0;

    int              rank = -1, ws = 0;
    bool             ok   = false;
};

static P2PState g;

/* ════════════════════════ Alloc / IPC ════════════════════════ */

torch::Tensor alloc_p2p_buffer(int64_t numel, int dev) {
    c10::cuda::CUDAGuard guard(dev);
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, numel * 2));
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, dev);
    return torch::from_blob(ptr, {numel}, [](void* p){ cudaFree(p); }, opts);
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

/* ════════════════════════ Init / Cleanup ════════════════════════ */

void init_p2p(torch::Tensor all_data_handles,
              torch::Tensor all_flag_handles,
              torch::Tensor local_tensor,
              int rank, int ws)
{
    TORCH_CHECK(!g.ok);
    TORCH_CHECK(ws <= MAX_GPUS);
    g.rank = rank;  g.ws = ws;  g.phase = 0;

    // for (int r = 0; r < ws; r++) {
    //     if (r == rank) continue;
    //     int can = 0;
    //     cudaDeviceCanAccessPeer(&can, rank, r);
    //     if (can) {
    //         cudaError_t e = cudaDeviceEnablePeerAccess(r, 0);
    //         if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    //         else if (e != cudaSuccess) CUDA_CHECK(e);
    //     }
    // }

    for (int r = 0; r < ws; r++) {
        if (r == rank) {
            g.bases[r] = nullptr;
            g.ptrs[r]  = local_tensor.data_ptr();
        } else {
            cudaIpcMemHandle_t h;
            std::memcpy(&h, all_data_handles[r].data_ptr(), sizeof(h));
            void* base = nullptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&base, h, cudaIpcMemLazyEnablePeerAccess));
            g.bases[r] = base;
            g.ptrs[r]  = base;
        }
    }
    CUDA_CHECK(cudaMalloc(&g.d_ptrs, ws * sizeof(__nv_bfloat16*)));
    CUDA_CHECK(cudaMemcpy(g.d_ptrs, g.ptrs, ws * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaMemcpy(g.d_flag_ptrs, g.flag_ptrs, ws * sizeof(int*), cudaMemcpyHostToDevice));

    // Explicitly disable peer access — forces IPC reads to fall back to PCIe staging
    for (int r = 0; r < ws; r++) {
        if (r == rank) continue;
        cudaDeviceDisablePeerAccess(r);
        cudaGetLastError();
    }

    g.ok = true;
}

void enable_peer_access() {
    TORCH_CHECK(g.ok);
    for (int r = 0; r < g.ws; r++) {
        if (r == g.rank) continue;
        int can = 0;
        cudaDeviceCanAccessPeer(&can, g.rank, r);
        if (can) {
            cudaError_t e = cudaDeviceEnablePeerAccess(r, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else if (e != cudaSuccess) CUDA_CHECK(e);
        }
    }
}

void disable_peer_access() {
    TORCH_CHECK(g.ok);
    for (int r = 0; r < g.ws; r++) {
        if (r == g.rank) continue;
        int can = 0;
        cudaDeviceCanAccessPeer(&can, g.rank, r);
        if (can) {
            cudaError_t e = cudaDeviceDisablePeerAccess(r);
            if (e == cudaErrorPeerAccessNotEnabled) cudaGetLastError();
            else if (e != cudaSuccess) CUDA_CHECK(e);
        }
    }
}

void cleanup_p2p() {
    if (!g.ok) return;
    for (int r = 0; r < g.ws; r++) {
        if (r != g.rank && g.bases[r])      cudaIpcCloseMemHandle(g.bases[r]);
        if (r != g.rank && g.flag_bases[r]) cudaIpcCloseMemHandle(g.flag_bases[r]);
    }
    if (g.d_ptrs)      cudaFree(g.d_ptrs);
    if (g.d_flag_ptrs)  cudaFree(g.d_flag_ptrs);
    if (g.local_flag)   cudaFree(g.local_flag);
    g = P2PState{};
}

/* ════════════════════════ Kernels ════════════════════════ */

// GPU-side barrier: spin until all flags >= phase
__global__ void barrier_kernel(int** flag_ptrs, int rank, int ws, int phase) {
    __threadfence_system();
    atomicExch(flag_ptrs[rank], phase);
    for (int r = 0; r < ws; r++) {
        if (r == rank) continue;
        while (atomicAdd(flag_ptrs[r], 0) < phase) {}
    }
    __threadfence_system();
}

// RS: read ONE source GPU's chunk, accumulate into local.
// 16 bf16 (32 bytes) per thread, two int4 loads.
__global__ void __launch_bounds__(256)
rs_one_source(__nv_bfloat16* __restrict__ data,
              __nv_bfloat16** __restrict__ all_ptrs,
              int src, int off, int elems)
{
    constexpr int V = 16;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= elems) return;
    int g = off + base;

    // Load remote
    int4 r0 = *reinterpret_cast<const int4*>(all_ptrs[src] + g);
    int4 r1 = *reinterpret_cast<const int4*>(all_ptrs[src] + g + 8);
    // Load local
    int4 l0 = *reinterpret_cast<const int4*>(data + g);
    int4 l1 = *reinterpret_cast<const int4*>(data + g + 8);

    const __nv_bfloat16* ra = reinterpret_cast<const __nv_bfloat16*>(&r0);
    const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&r1);
    const __nv_bfloat16* la = reinterpret_cast<const __nv_bfloat16*>(&l0);
    const __nv_bfloat16* lb = reinterpret_cast<const __nv_bfloat16*>(&l1);

    int4 o0, o1;
    __nv_bfloat16* oa = reinterpret_cast<__nv_bfloat16*>(&o0);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o1);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        oa[i] = __float2bfloat16(__bfloat162float(ra[i]) + __bfloat162float(la[i]));
    #pragma unroll
    for (int i = 0; i < 8; i++)
        ob[i] = __float2bfloat16(__bfloat162float(rb[i]) + __bfloat162float(lb[i]));

    *reinterpret_cast<int4*>(data + g)     = o0;
    *reinterpret_cast<int4*>(data + g + 8) = o1;
}

// AG: copy ONE source rank's reduced chunk to local.
__global__ void __launch_bounds__(256)
ag_one_source(__nv_bfloat16* __restrict__ data,
              __nv_bfloat16** __restrict__ all_ptrs,
              int src, int off, int elems)
{
    constexpr int V = 16;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= elems) return;
    int g = off + base;

    int4 v0 = *reinterpret_cast<const int4*>(all_ptrs[src] + g);
    int4 v1 = *reinterpret_cast<const int4*>(all_ptrs[src] + g + 8);
    *reinterpret_cast<int4*>(data + g)     = v0;
    *reinterpret_cast<int4*>(data + g + 8) = v1;
}

/* ════════════════════════ Launch ════════════════════════ */

void fused_allreduce(torch::Tensor t, int64_t n) {
    TORCH_CHECK(g.ok);
    int ws = g.ws, rank = g.rank;
    int chunk = n / ws;
    int my_off = rank * chunk;
    auto* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr());

    constexpr int B = 256, V = 16;
    int rs_grid = (chunk / V + B - 1) / B;

    // ── ReduceScatter: read from one source at a time, accumulate ──
    // STAGGERED order: rank r starts from source (r+1)%ws, so at any step
    // each source GPU is read by exactly ONE rank → zero NVLink contention.
    for (int i = 0; i < ws - 1; i++) {
        int s = (rank + 1 + i) % ws;
        rs_one_source<<<rs_grid, B>>>(p, g.d_ptrs, s, my_off, chunk);
    }

    // ── Barrier: all ranks must finish RS before AG reads their chunks ──
    barrier_kernel<<<1, 1>>>(g.d_flag_ptrs, rank, ws, ++g.phase);

    // ── AllGather: read each remote rank's reduced chunk ──
    // Same staggered pattern for AG.
    for (int i = 0; i < ws - 1; i++) {
        int s = (rank + 1 + i) % ws;
        int off = s * chunk;
        ag_one_source<<<rs_grid, B>>>(p, g.d_ptrs, s, off, chunk);
    }

    // ── Barrier: safe for next iteration ──
    barrier_kernel<<<1, 1>>>(g.d_flag_ptrs, rank, ws, ++g.phase);
}

/* ════════════════════════ pybind ════════════════════════ */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc_p2p_buffer",  &alloc_p2p_buffer);
    m.def("get_ipc_handle",    &get_ipc_handle);
    m.def("alloc_flag_buffer", &alloc_flag_buffer);
    m.def("init_p2p",          &init_p2p);
    m.def("cleanup_p2p",         &cleanup_p2p);
    m.def("enable_peer_access",  &enable_peer_access);
    m.def("disable_peer_access", &disable_peer_access);
    m.def("fused_allreduce",     &fused_allreduce);
}
