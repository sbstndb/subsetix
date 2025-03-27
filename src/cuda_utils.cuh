#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Tjhis code is not mine

// Returns the error code if one occurs, allowing the caller to handle it.
// Use like: err = CUDA_CHECK(cudaMalloc(...)); if (err != cudaSuccess) return err;
#define CUDA_CHECK(call) [&]() -> cudaError_t { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA API Error in %s:%d - %s: %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
    } \
    return err; \
}()

// Returns cudaSuccess or the specific error code.
// Use like: err = KERNEL_CHECK(); if (err != cudaSuccess) { /* cleanup */ return err; }
#define KERNEL_CHECK() [&]() -> cudaError_t { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Kernel Launch Error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Synchronization Error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
    return cudaSuccess; \
}()

#endif // CUDA_UTILS_CUH
