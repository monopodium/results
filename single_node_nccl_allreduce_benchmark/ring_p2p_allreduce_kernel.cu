/*
 * Ring AllReduce via CUDA IPC — optimised for NVSwitch fully-connected topology.
 *
 * Each rank reads only from its LEFT neighbour.
 * GPU-side atomic flag sync — zero CPU barriers.
 *
 * v2: one compute kernel per step (no sub-step pipes).
 *     Signal fused into compute kernel via last-block atomic.
 *     Total kernel launches = 2*(N-1) + 1 = 15 for 8 GPUs.
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

struct RingState {
    __nv_bfloat16*  left_ptr       = nullptr;
    void*           left_base      = nullptr;

    int*            my_flag        = nullptr;
    int*            left_flag      = nullptr;
    void*           left_flag_base = nullptr;

    int             rank = -1, ws = 0, left = -1;
    bool            ok = false;
};

static RingState rg;

/* ════════ device-side block counter (for last-block signal) ════════ */
__device__ unsigned int d_block_done = 0;

/* ════════════════════════ Alloc / IPC ════════════════════════ */

torch::Tensor ring_alloc_buffer(int64_t numel, int dev) {
    c10::cuda::CUDAGuard guard(dev);
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, numel * 2));
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, dev);
    return torch::from_blob(p, {numel}, [](void* x){ cudaFree(x); }, opts);
}

torch::Tensor ring_get_handle(torch::Tensor t) {
    cudaIpcMemHandle_t h;
    CUDA_CHECK(cudaIpcGetMemHandle(&h, t.data_ptr()));
    auto out = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
    std::memcpy(out.data_ptr(), &h, sizeof(h));
    return out;
}

torch::Tensor ring_alloc_flag() {
    int* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, sizeof(int)));
    CUDA_CHECK(cudaMemset(p, 0, sizeof(int)));
    rg.my_flag = p;
    cudaIpcMemHandle_t h;
    CUDA_CHECK(cudaIpcGetMemHandle(&h, p));
    auto out = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
    std::memcpy(out.data_ptr(), &h, sizeof(h));
    return out;
}

/* ════════════════════════ Init / Cleanup ════════════════════════ */

void ring_init(torch::Tensor all_data_handles,
               torch::Tensor all_flag_handles,
               torch::Tensor local_tensor,
               int rank, int ws)
{
    TORCH_CHECK(!rg.ok);
    rg.rank = rank;  rg.ws = ws;  rg.left = (rank - 1 + ws) % ws;

    for (int r = 0; r < ws; r++) {
        if (r == rank) continue;
        int can = 0;
        cudaDeviceCanAccessPeer(&can, rank, r);
        if (can) {
            cudaError_t e = cudaDeviceEnablePeerAccess(r, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else if (e != cudaSuccess) CUDA_CHECK(e);
        }
    }

    {
        cudaIpcMemHandle_t h;
        std::memcpy(&h, all_data_handles[rg.left].data_ptr(), sizeof(h));
        void* base = nullptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&base, h, cudaIpcMemLazyEnablePeerAccess));
        rg.left_base = base;
        rg.left_ptr  = static_cast<__nv_bfloat16*>(base);
    }
    {
        cudaIpcMemHandle_t h;
        std::memcpy(&h, all_flag_handles[rg.left].data_ptr(), sizeof(h));
        void* base = nullptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&base, h, cudaIpcMemLazyEnablePeerAccess));
        rg.left_flag_base = base;
        rg.left_flag      = static_cast<int*>(base);
    }
    rg.ok = true;
}

void ring_cleanup() {
    if (!rg.ok) return;
    if (rg.left_base)      cudaIpcCloseMemHandle(rg.left_base);
    if (rg.left_flag_base) cudaIpcCloseMemHandle(rg.left_flag_base);
    if (rg.my_flag)        cudaFree(rg.my_flag);
    rg = RingState{};
}

/* ════════════════════════ Kernels ════════════════════════ */

// 1-thread: spin until left flag >= val
__global__ void wait_flag(int* left_flag, int val) {
    while (atomicAdd(left_flag, 0) < val) {}
    __threadfence_system();
}

// RS: read left's chunk, accumulate.  Last block signals my_flag.
__global__ void __launch_bounds__(256)
rs_kernel_fused(__nv_bfloat16* __restrict__ data,
                const __nv_bfloat16* __restrict__ left,
                int off, int chunk,
                int* my_flag, int signal_val)
{
    constexpr int V = 16;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= chunk) return;

    int g = off + base;

    int4 a0 = *reinterpret_cast<const int4*>(left + g);
    int4 a1 = *reinterpret_cast<const int4*>(left + g + 8);
    int4 b0 = *reinterpret_cast<const int4*>(data + g);
    int4 b1 = *reinterpret_cast<const int4*>(data + g + 8);

    const __nv_bfloat16* la0 = reinterpret_cast<const __nv_bfloat16*>(&a0);
    const __nv_bfloat16* la1 = reinterpret_cast<const __nv_bfloat16*>(&a1);
    const __nv_bfloat16* lb0 = reinterpret_cast<const __nv_bfloat16*>(&b0);
    const __nv_bfloat16* lb1 = reinterpret_cast<const __nv_bfloat16*>(&b1);

    int4 r0, r1;
    __nv_bfloat16* ro0 = reinterpret_cast<__nv_bfloat16*>(&r0);
    __nv_bfloat16* ro1 = reinterpret_cast<__nv_bfloat16*>(&r1);
    #pragma unroll
    for (int i = 0; i < 8; i++)
        ro0[i] = __float2bfloat16(__bfloat162float(la0[i]) + __bfloat162float(lb0[i]));
    #pragma unroll
    for (int i = 0; i < 8; i++)
        ro1[i] = __float2bfloat16(__bfloat162float(la1[i]) + __bfloat162float(lb1[i]));

    *reinterpret_cast<int4*>(data + g)     = r0;
    *reinterpret_cast<int4*>(data + g + 8) = r1;

    // Last-block signal
    __threadfence_system();
    if (threadIdx.x == 0) {
        unsigned int old = atomicAdd(&d_block_done, 1);
        if (old == gridDim.x - 1) {
            d_block_done = 0;
            atomicExch(my_flag, signal_val);
        }
    }
}

// AG: read left's chunk, copy.  Last block signals.
__global__ void __launch_bounds__(256)
ag_kernel_fused(__nv_bfloat16* __restrict__ data,
                const __nv_bfloat16* __restrict__ left,
                int off, int chunk,
                int* my_flag, int signal_val)
{
    constexpr int V = 16;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= chunk) return;

    int g = off + base;
    int4 v0 = *reinterpret_cast<const int4*>(left + g);
    int4 v1 = *reinterpret_cast<const int4*>(left + g + 8);
    *reinterpret_cast<int4*>(data + g)     = v0;
    *reinterpret_cast<int4*>(data + g + 8) = v1;

    // Last-block signal
    __threadfence_system();
    if (threadIdx.x == 0) {
        unsigned int old = atomicAdd(&d_block_done, 1);
        if (old == gridDim.x - 1) {
            d_block_done = 0;
            atomicExch(my_flag, signal_val);
        }
    }
}

/* ════════════════════════ Launch ════════════════════════ */

void ring_allreduce(torch::Tensor t, int64_t n, int /*num_pipes — ignored*/) {
    TORCH_CHECK(rg.ok);
    int ws   = rg.ws, rank = rg.rank;
    int chunk = n / ws;
    auto* data = reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
    const auto* left = rg.left_ptr;

    constexpr int B = 256, V = 16;
    int grid = (chunk / V + B - 1) / B;
    int counter = 0;

    // ── ReduceScatter ──
    for (int step = 0; step < ws - 1; step++) {
        int recv_idx = (rank - step - 1 + ws) % ws;
        int off = recv_idx * chunk;

        if (step > 0)
            wait_flag<<<1, 1>>>(rg.left_flag, counter);

        counter++;
        rs_kernel_fused<<<grid, B>>>(data, left, off, chunk,
                                     rg.my_flag, counter);
    }

    // ── AllGather ──
    for (int step = 0; step < ws - 1; step++) {
        int recv_idx = (rank - step + ws) % ws;
        int off = recv_idx * chunk;

        // AG always waits (left must have finished corresponding step)
        wait_flag<<<1, 1>>>(rg.left_flag, counter);

        counter++;
        ag_kernel_fused<<<grid, B>>>(data, left, off, chunk,
                                     rg.my_flag, counter);
    }

    // Final wait: ensure left finished everything
    wait_flag<<<1, 1>>>(rg.left_flag, counter);
}

/* ════════════════════════ pybind ════════════════════════ */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc_buffer", &ring_alloc_buffer);
    m.def("get_handle",   &ring_get_handle);
    m.def("alloc_flag",   &ring_alloc_flag);
    m.def("init",         &ring_init);
    m.def("cleanup",      &ring_cleanup);
    m.def("allreduce",    &ring_allreduce);
}
