#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "common/enumerations.h"

struct NetCalcKernelParams
{
    void* inputs;
    void* weighs;
    void* results;
    unsigned input_size;
    unsigned output_size;
};

// 16 KB shared memory, 16 bytes for blockIdx, blockDim y gridDim
#define CUDA_MAX_SHARED_SIZE ( 16384 - ( 16 + sizeof(NetCalcKernelParams) ) )
// TODO intentar quitar estas dos constantes:
#define CUDA_MAX_SHARED_FLOATS ( CUDA_MAX_SHARED_SIZE / 4 )
#define CUDA_MAX_SHARED_BITS ( CUDA_MAX_SHARED_FLOATS * BITS_PER_UNSIGNED )

static unsigned Cuda_Threads_Per_Block = 256;

// cuda_basic.cu
void checkCUDAError(const char *msg);

extern "C" void* cuda_malloc(unsigned byteSize);
extern "C" void cuda_free(void* d_ptr);
extern "C" void cuda_copyToDevice(void* d_dest, void* h_src, unsigned count);
extern "C" void cuda_copyToHost(void* h_dest, void* d_src, unsigned count);
extern "C" void cuda_setZero(void* data, unsigned byteSize, BufferType bufferType, unsigned block_size);

// cuda_activation.cu
extern "C" void cuda_activation(void* data, unsigned size, BufferType bufferType, float* results,
                                float* thresholds, FunctionType functionType, unsigned block_size);

// cuda_genetic.cu
extern "C" void cuda_mutateWeigh(void* buffer, unsigned pos, float mutation, BufferType bufferType);
extern "C" void cuda_resetWeigh(void* buffer, unsigned pos, BufferType bufferType);
extern "C" void cuda_crossover(void* buffer1, void* buffer2, unsigned* bitBuffer, unsigned size,
                               BufferType bufferType, unsigned block_size);

// cuda_reduction0.cu
extern "C" void cuda_netCalcReduction0(BufferType inputType, unsigned block_size, void* inputPtr,
                                       void* weighs, float* results, unsigned input_size,
                                       unsigned output_size);
// cuda_reduction.cu
extern "C" void
        cuda_netCalcReduction(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                              float* results, unsigned input_size, unsigned output_size);
//
extern "C" void cuda_netCalcOutputs(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                                    float* results, unsigned input_size, unsigned output_size);
//
extern "C" void
        cuda_netCalcInvMatrix(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                              float* results, unsigned input_size, unsigned output_size);

#endif /*CUDA_DEFINITIONS_H_*/
