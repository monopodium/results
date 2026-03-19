#include <cuda.h>
#include <stdio.h>

#define CHECK(call) \
{ \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
}

__device__ __forceinline__ int get_smid() {
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

__global__ void testKernel() {
    int smid = get_smid();

    if (threadIdx.x == 0) {
        printf("Block %d running on SM %d\n", blockIdx.x, smid);
    }
}

int main() {

    CHECK(cuInit(0));

    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    int totalSM;
    CHECK(cuDeviceGetAttribute(
        &totalSM,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        dev));

    printf("GPU has %d SMs\n", totalSM);

    // Step 1: Get the device's SM resources
    CUdevResource devResource;
    CHECK(cuDeviceGetDevResource(dev, &devResource, CU_DEV_RESOURCE_TYPE_SM));
    printf("Device has %u SMs available\n", devResource.sm.smCount);

    // Step 2: Split SM resources - request half the SMs
    unsigned int minCount = totalSM / 2;
    // Round to multiple of 8 for SM 9.0+ (H200)
    minCount = (minCount / 8) * 8;
    printf("Requesting green context with %u SMs\n", minCount);

    unsigned int nbGroups = 1;
    CUdevResource splitResult;
    CUdevResource remaining;
    CHECK(cuDevSmResourceSplitByCount(
        &splitResult, &nbGroups,
        &devResource, &remaining,
        0, minCount));

    printf("Got partition with %u SMs (remaining: %u SMs)\n",
           splitResult.sm.smCount, remaining.sm.smCount);

    // Step 3: Generate resource descriptor
    CUdevResourceDesc resDesc;
    CHECK(cuDevResourceGenerateDesc(&resDesc, &splitResult, 1));

    // Step 4: Create green context
    CUgreenCtx greenCtx;
    CHECK(cuGreenCtxCreate(&greenCtx, resDesc, dev, CU_GREEN_CTX_DEFAULT_STREAM));

    // Convert green context to a regular CUcontext for kernel launch
    CUcontext ctx;
    CHECK(cuCtxFromGreenCtx(&ctx, greenCtx));

    // Push this context as current
    CHECK(cuCtxPushCurrent(ctx));

    printf("Green context created and set as current\n");

    testKernel<<<64, 256>>>();

    cudaDeviceSynchronize();

    printf("Kernel finished\n");

    CUcontext popped;
    cuCtxPopCurrent(&popped);
    cuGreenCtxDestroy(greenCtx);

    return 0;
}
