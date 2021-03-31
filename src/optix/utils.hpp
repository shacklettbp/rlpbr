#pragma once

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <optix.h>

namespace RLpbr {
namespace optix {

void printOptixError(OptixResult res, const char *msg);

static inline OptixResult checkOptix(OptixResult res, const char *msg,
                                     bool fatal = true) noexcept
{
    if (res != OPTIX_SUCCESS) {
        printOptixError(res, msg);
        if (fatal) {
            std::abort();
        }
    }

    return res;
}

void printCudaError(cudaError_t res, const char *msg);

static inline cudaError_t checkCuda(cudaError_t res, const char *msg,
                                    bool fatal = true) noexcept
{
    if (res != cudaSuccess) {
        printCudaError(res, msg);
        if (fatal) {
            std::abort();
        }
    }

    return res;
}

void printNVRTCError(nvrtcResult res, const char *msg);

static inline nvrtcResult checkNVRTC(nvrtcResult res, const char *msg,
                                     bool fatal = true) noexcept
{
    if (res != NVRTC_SUCCESS) {
        printNVRTCError(res, msg);
        if (fatal) {
            std::abort();
        }
    }

    return res;
}

#define STRINGIFY_HELPER(m) #m
#define STRINGIFY(m) STRINGIFY_HELPER(m)

#define LOC_APPEND(m) m ": " __FILE__ " @ " STRINGIFY(__LINE__)
#define REQ_OPTIX(expr) checkOptix((expr), LOC_APPEND(#expr))
#define CHK_OPTIX(expr) checkOptix((expr), LOC_APPEND(#expr), false)

#define REQ_CUDA(expr) checkCuda((expr), LOC_APPEND(#expr))
#define CHK_CUDA(expr) checkCuda((expr), LOC_APPEND(#expr), false)

#define REQ_NVRTC(expr) checkNVRTC((expr), LOC_APPEND(#expr))
#define CHK_NVRTC(expr) checkNVRTC((expr), LOC_APPEND(#expr), false)

inline void *allocCU(size_t num_bytes)
{
    void *ptr;
    REQ_CUDA(cudaMalloc(&ptr, num_bytes));

    return ptr;
}

inline void *allocCUHost(size_t num_bytes, int flags = 0)
{
    void *ptr;
    REQ_CUDA(cudaHostAlloc(&ptr, num_bytes, flags));

    return ptr;
}

}
}
