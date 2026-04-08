/*
 * Two-shot AllReduce via cudaMemcpyAsync with host-staging pipeline.
 *
 * Host-staging mode uses double-buffered host pinned memory + 2 CUDA streams
 * to overlap D2H and H2D transfers (PCIe is full-duplex):
 *
 *   Stream A (D2H):  [pipe0 D2H] [pipe1 D2H] [pipe2 D2H] ...
 *   Stream B (H2D):            [pipe0 H2D+accum] [pipe1 H2D+accum] ...
 *
 * This roughly doubles effective bandwidth vs serial D2H→H2D.
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

struct McpyState {
    void*  bases[MAX_GPUS] = {};
    void*  ptrs[MAX_GPUS]  = {};

    int*   local_flag      = nullptr;
    void*  flag_bases[MAX_GPUS] = {};
    int*   flag_ptrs[MAX_GPUS]  = {};
    int**  d_flag_ptrs     = nullptr;
    int    phase           = 0;

    /* staging buffers */
    __nv_bfloat16* local_buf     = nullptr;   /* GPU buf for accum */
    __nv_bfloat16* host_buf[2]   = {};        /* double-buffered host pinned */
    cudaStream_t   st_d2h        = nullptr;   /* stream for D2H */
    cudaStream_t   st_h2d        = nullptr;   /* stream for H2D + accum */
    cudaEvent_t    ev[2]         = {};        /* sync events */

    bool   use_host_staging = false;
    int    num_pipes        = 1;
    int    rank = -1, ws = 0;
    bool   ok   = false;
};

static McpyState g;

/* ════════════════ Alloc / IPC ════════════════ */

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

/* ════════════════ Init / Cleanup ════════════════ */

void init_p2p(torch::Tensor all_data_handles,
              torch::Tensor all_flag_handles,
              torch::Tensor local_tensor,
              int rank, int ws, int64_t max_chunk_elems,
              bool use_host_staging, int num_pipes)
{
    TORCH_CHECK(!g.ok);
    TORCH_CHECK(ws <= MAX_GPUS);
    g.rank = rank;  g.ws = ws;  g.phase = 0;
    g.use_host_staging = use_host_staging;
    g.num_pipes = num_pipes;

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

    /* Local GPU staging buffer (one full chunk for accum) */
    CUDA_CHECK(cudaMalloc(&g.local_buf, max_chunk_elems * sizeof(__nv_bfloat16)));

    if (use_host_staging) {
        /* Double-buffered host pinned, each holds one pipe = chunk/num_pipes */
        size_t pipe_bytes = (max_chunk_elems / num_pipes) * sizeof(__nv_bfloat16);
        CUDA_CHECK(cudaMallocHost(&g.host_buf[0], pipe_bytes));
        CUDA_CHECK(cudaMallocHost(&g.host_buf[1], pipe_bytes));
        CUDA_CHECK(cudaStreamCreate(&g.st_d2h));
        CUDA_CHECK(cudaStreamCreate(&g.st_h2d));
        CUDA_CHECK(cudaEventCreate(&g.ev[0]));
        CUDA_CHECK(cudaEventCreate(&g.ev[1]));
    }

    g.ok = true;
}

void cleanup_p2p() {
    if (!g.ok) return;
    for (int r = 0; r < g.ws; r++) {
        if (r != g.rank && g.bases[r])      cudaIpcCloseMemHandle(g.bases[r]);
        if (r != g.rank && g.flag_bases[r]) cudaIpcCloseMemHandle(g.flag_bases[r]);
    }
    if (g.d_flag_ptrs)  cudaFree(g.d_flag_ptrs);
    if (g.local_flag)   cudaFree(g.local_flag);
    if (g.local_buf)    cudaFree(g.local_buf);
    if (g.host_buf[0])  cudaFreeHost(g.host_buf[0]);
    if (g.host_buf[1])  cudaFreeHost(g.host_buf[1]);
    if (g.st_d2h)       cudaStreamDestroy(g.st_d2h);
    if (g.st_h2d)       cudaStreamDestroy(g.st_h2d);
    if (g.ev[0])        cudaEventDestroy(g.ev[0]);
    if (g.ev[1])        cudaEventDestroy(g.ev[1]);
    g = McpyState{};
}

/* ════════════════ Kernels ════════════════ */

__global__ void barrier_kernel(int** flag_ptrs, int rank, int ws, int phase) {
    __threadfence_system();
    atomicExch(flag_ptrs[rank], phase);
    for (int r = 0; r < ws; r++) {
        if (r == rank) continue;
        while (atomicAdd(flag_ptrs[r], 0) < phase) {}
    }
    __threadfence_system();
}

/* Accumulate: data[off..off+n] += buf[0..n]  (bf16x2 via __hadd2) */
__global__ void __launch_bounds__(256)
accum_local(__nv_bfloat16* __restrict__ data,
            const __nv_bfloat16* __restrict__ buf,
            int off, int n)
{
    constexpr int V = 16;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * V;
    if (base >= n) return;
    int idx = off + base;

    int4 l0 = *reinterpret_cast<const int4*>(data + idx);
    int4 l1 = *reinterpret_cast<const int4*>(data + idx + 8);
    int4 b0 = *reinterpret_cast<const int4*>(buf + base);
    int4 b1 = *reinterpret_cast<const int4*>(buf + base + 8);

    int4 o0, o1;
    __nv_bfloat162* o2a = reinterpret_cast<__nv_bfloat162*>(&o0);
    __nv_bfloat162* o2b = reinterpret_cast<__nv_bfloat162*>(&o1);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        o2a[i] = __hadd2(reinterpret_cast<__nv_bfloat162*>(&l0)[i],
                         reinterpret_cast<__nv_bfloat162*>(&b0)[i]);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        o2b[i] = __hadd2(reinterpret_cast<__nv_bfloat162*>(&l1)[i],
                         reinterpret_cast<__nv_bfloat162*>(&b1)[i]);

    *reinterpret_cast<int4*>(data + idx)     = o0;
    *reinterpret_cast<int4*>(data + idx + 8) = o1;
}

/* ════════════════ Pipelined host-staging ════════════════ */

/*
 * RS for one remote source: pipelined D2H / H2D+accum.
 *
 *   st_d2h:  [D2H pipe0] [D2H pipe1] [D2H pipe2] ...
 *   st_h2d:      [wait]  [H2D+acc 0] [wait] [H2D+acc 1] ...
 *
 * Double-buffered host pinned memory (host_buf[0], host_buf[1]).
 */
static void rs_one_source_staged(
    __nv_bfloat16* data, int my_off, int chunk,
    const __nv_bfloat16* remote, int num_pipes)
{
    int pipe = chunk / num_pipes;
    size_t pipe_bytes = pipe * sizeof(__nv_bfloat16);

    constexpr int B = 256, V = 16;
    int grid = (pipe / V + B - 1) / B;

    for (int p = 0; p < num_pipes; p++) {
        int buf = p & 1;
        int off = p * pipe;

        /* D2H: remote GPU → host pinned buf */
        CUDA_CHECK(cudaMemcpyAsync(
            g.host_buf[buf], remote + my_off + off, pipe_bytes,
            cudaMemcpyDeviceToHost, g.st_d2h));
        CUDA_CHECK(cudaEventRecord(g.ev[buf], g.st_d2h));

        /* H2D + accum: wait for D2H, copy to local_buf, accumulate */
        CUDA_CHECK(cudaStreamWaitEvent(g.st_h2d, g.ev[buf]));
        CUDA_CHECK(cudaMemcpyAsync(
            g.local_buf, g.host_buf[buf], pipe_bytes,
            cudaMemcpyHostToDevice, g.st_h2d));
        accum_local<<<grid, B, 0, g.st_h2d>>>(data, g.local_buf, my_off + off, pipe);
    }
}

/*
 * AG for one remote source: pipelined D2H / H2D copy.
 */
static void ag_one_source_staged(
    __nv_bfloat16* data, int src_off, int chunk,
    const __nv_bfloat16* remote, int num_pipes)
{
    int pipe = chunk / num_pipes;
    size_t pipe_bytes = pipe * sizeof(__nv_bfloat16);

    for (int p = 0; p < num_pipes; p++) {
        int buf = p & 1;
        int off = p * pipe;

        CUDA_CHECK(cudaMemcpyAsync(
            g.host_buf[buf], remote + src_off + off, pipe_bytes,
            cudaMemcpyDeviceToHost, g.st_d2h));
        CUDA_CHECK(cudaEventRecord(g.ev[buf], g.st_d2h));

        CUDA_CHECK(cudaStreamWaitEvent(g.st_h2d, g.ev[buf]));
        CUDA_CHECK(cudaMemcpyAsync(
            data + src_off + off, g.host_buf[buf], pipe_bytes,
            cudaMemcpyHostToDevice, g.st_h2d));
    }
}

/* Drain pipeline: sync st_h2d back to main stream */
static void drain_pipeline(cudaStream_t main_st) {
    CUDA_CHECK(cudaEventRecord(g.ev[0], g.st_h2d));
    CUDA_CHECK(cudaStreamWaitEvent(main_st, g.ev[0]));
}

/* ════════════════ Launch ════════════════ */

void fused_allreduce(torch::Tensor t, int64_t n) {
    TORCH_CHECK(g.ok);
    int ws = g.ws, rank = g.rank;
    int chunk = n / ws;
    int my_off = rank * chunk;
    auto* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
    size_t chunk_bytes = chunk * sizeof(__nv_bfloat16);
    cudaStream_t main_st = at::cuda::getCurrentCUDAStream();

    constexpr int B = 256, V = 16;
    int grid = (chunk / V + B - 1) / B;

    if (!g.use_host_staging) {
        /* ── D2D mode (NVLink via IPC) ── */
        for (int i = 0; i < ws - 1; i++) {
            int s = (rank + 1 + i) % ws;
            auto* remote = static_cast<__nv_bfloat16*>(g.ptrs[s]);
            CUDA_CHECK(cudaMemcpyAsync(g.local_buf, remote + my_off,
                                       chunk_bytes, cudaMemcpyDeviceToDevice, main_st));
            accum_local<<<grid, B, 0, main_st>>>(p, g.local_buf, my_off, chunk);
        }
        barrier_kernel<<<1, 1, 0, main_st>>>(g.d_flag_ptrs, rank, ws, ++g.phase);

        for (int i = 0; i < ws - 1; i++) {
            int s = (rank + 1 + i) % ws;
            auto* remote = static_cast<__nv_bfloat16*>(g.ptrs[s]);
            CUDA_CHECK(cudaMemcpyAsync(p + s * chunk, remote + s * chunk,
                                       chunk_bytes, cudaMemcpyDeviceToDevice, main_st));
        }
        barrier_kernel<<<1, 1, 0, main_st>>>(g.d_flag_ptrs, rank, ws, ++g.phase);

    } else {
        /* ── Host-staged mode (PCIe, pipelined D2H/H2D) ── */
        int np = g.num_pipes;

        /* Ensure pipeline streams start after main stream work */
        CUDA_CHECK(cudaEventRecord(g.ev[0], main_st));
        CUDA_CHECK(cudaStreamWaitEvent(g.st_d2h, g.ev[0]));
        CUDA_CHECK(cudaStreamWaitEvent(g.st_h2d, g.ev[0]));

        /* ── ReduceScatter ── */
        for (int i = 0; i < ws - 1; i++) {
            int s = (rank + 1 + i) % ws;
            auto* remote = static_cast<__nv_bfloat16*>(g.ptrs[s]);
            rs_one_source_staged(p, my_off, chunk, remote, np);
        }
        drain_pipeline(main_st);

        barrier_kernel<<<1, 1, 0, main_st>>>(g.d_flag_ptrs, rank, ws, ++g.phase);

        /* Sync pipeline streams to barrier completion */
        CUDA_CHECK(cudaEventRecord(g.ev[0], main_st));
        CUDA_CHECK(cudaStreamWaitEvent(g.st_d2h, g.ev[0]));
        CUDA_CHECK(cudaStreamWaitEvent(g.st_h2d, g.ev[0]));

        /* ── AllGather ── */
        for (int i = 0; i < ws - 1; i++) {
            int s = (rank + 1 + i) % ws;
            auto* remote = static_cast<__nv_bfloat16*>(g.ptrs[s]);
            ag_one_source_staged(p, s * chunk, chunk, remote, np);
        }
        drain_pipeline(main_st);

        barrier_kernel<<<1, 1, 0, main_st>>>(g.d_flag_ptrs, rank, ws, ++g.phase);
    }
}

/* ════════════════ pybind ════════════════ */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alloc_p2p_buffer",  &alloc_p2p_buffer);
    m.def("get_ipc_handle",    &get_ipc_handle);
    m.def("alloc_flag_buffer", &alloc_flag_buffer);
    m.def("init_p2p",          &init_p2p);
    m.def("cleanup_p2p",       &cleanup_p2p);
    m.def("fused_allreduce",   &fused_allreduce);
}
