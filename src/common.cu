#ifndef COMMON
#define COMMON

#include <cstdio>
#include <thrust/random.h>
#include "intersections.h"
#include <thrust/device_ptr.h>

#define ERRORCHECK 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define dPtr(x) thrust::device_pointer_cast(x)

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaError_t err = cudaDeviceSynchronize();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

#define CREATE_RANDOM_ENGINE(iter, idx, depth, u01, rng) \
thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth); \
thrust::uniform_real_distribution<float> u01(0, 1)


#endif COMMON