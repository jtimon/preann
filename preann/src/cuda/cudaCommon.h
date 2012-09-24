#ifndef CUDA_COMMON_DEFINITIONS_H_
#define CUDA_COMMON_DEFINITIONS_H_

#include "cuda.h"

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

template<typename T>
inline __device__ T device_min(T a, T b)
{
    return (a < b) ? a : b;
}

#endif /*CUDA_COMMON_DEFINITIONS_H_*/
