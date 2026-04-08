#include <torch/extension.h>
#include <cuda_runtime.h>

void enable_peer(int src, int dst) {
    cudaSetDevice(src);
    cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
}

void disable_peer(int src, int dst) {
    cudaSetDevice(src);
    cudaError_t e = cudaDeviceDisablePeerAccess(dst);
    if (e == cudaErrorPeerAccessNotEnabled) cudaGetLastError();
}

int can_peer(int src, int dst) {
    int can = 0;
    cudaDeviceCanAccessPeer(&can, src, dst);
    return can;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enable_peer",  &enable_peer);
    m.def("disable_peer", &disable_peer);
    m.def("can_peer",     &can_peer);
}
