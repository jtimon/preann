#include "cudaCommon.h"

// ERRORS

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        printf("Cuda error: %s : %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// CLOCK

extern "C" void cuda_synchronize()
{
    cudaThreadSynchronize();
    checkCUDAError("synchronize");
}


// MEMORY MANAGEMENT

extern "C" void* cuda_malloc(unsigned byteSize)
{
    void* ptr;
    cudaMalloc((void**) &(ptr), byteSize);

    checkCUDAError("malloc");
    return ptr;
}

extern "C" void cuda_free(void* d_ptr)
{
    cudaFree(d_ptr);
    checkCUDAError("free");
}

extern "C" void cuda_copyToDevice(void* d_dest, void* h_src, unsigned count)
{
    cudaMemcpy(d_dest, h_src, count, cudaMemcpyHostToDevice);
    checkCUDAError("copyToDevice");
}

extern "C" void cuda_copyToHost(void* h_dest, void* d_src, unsigned count)
{
    cudaMemcpy(h_dest, d_src, count, cudaMemcpyDeviceToHost);
    checkCUDAError("copyToHost");
}

// INITIALIZATION

template <class bufferType>
__global__
void SetValueToAnArrayKernel(bufferType* data, unsigned size, bufferType value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    data[idx] = value;
}

extern "C" void cuda_setZero(void* data, unsigned byteSize, BufferType bufferType, unsigned block_size)
{
    unsigned grid_size;
    unsigned size;

    switch (bufferType) {
        case BT_BYTE:
            size = byteSize / sizeof(unsigned char);
            grid_size = ((size - 1) / block_size) + 1;
            SetValueToAnArrayKernel<unsigned char><<< grid_size, block_size >>>((unsigned char*)data, size, (unsigned char)0);
            break;
        case BT_FLOAT:
            size = byteSize / sizeof(float);
            grid_size = ((size - 1) / block_size) + 1;
            SetValueToAnArrayKernel<float><<< grid_size, block_size >>>((float*)data, size, 0);
            break;
        case BT_BIT:
        case BT_SIGN:
            cudaMemset(data, 0, byteSize);
            break;
    }
}

