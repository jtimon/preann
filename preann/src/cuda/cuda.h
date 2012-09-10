#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "common/enumerations.h"

// 16 KB shared memory, 16 bytes for blockIdx, blockDim y gridDim
// up to 256 for kernel params (20 bytes in our cases)
// But need aligned segments for coalescing access (64 bytes)
#define CUDA_MAX_SHARED_SIZE ( 16384 - 64 )

static unsigned Cuda_Threads_Per_Block = 256;

// basic.cu
void checkCUDAError(const char *msg);

extern "C" void* cuda_malloc(unsigned byteSize);
extern "C" void cuda_free(void* d_ptr);
extern "C" void cuda_copyToDevice(void* d_dest, void* h_src, unsigned count);
extern "C" void cuda_copyToHost(void* h_dest, void* d_src, unsigned count);
extern "C" void cuda_setZero(void* data, unsigned byteSize, BufferType bufferType, unsigned block_size);

// activation.cu
extern "C" void cuda_activation(void* data, unsigned size, BufferType bufferType, float* results,
                                float* thresholds, FunctionType functionType, unsigned block_size);

// genetic.cu
extern "C" void cuda_mutateWeigh(void* buffer, unsigned pos, float mutation, BufferType bufferType);
extern "C" void cuda_resetWeigh(void* buffer, unsigned pos, BufferType bufferType);
extern "C" void cuda_crossover(void* buffer1, void* buffer2, unsigned* bitBuffer, unsigned size, BufferType bufferType);
extern "C" void cuda_crossoverOld(void* buffer1, void* buffer2, unsigned* bitBuffer, unsigned size,
                               BufferType bufferType, unsigned block_size);

// calcReduction0.cu
extern "C" void cuda_netCalcReduction0(BufferType inputType, unsigned block_size, void* inputPtr,
                                       void* weighs, float* results, unsigned input_size,
                                       unsigned output_size);
// calcReduction.cu
extern "C" void
        cuda_netCalcReduction(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                              float* results, unsigned input_size, unsigned output_size);
// calcOutputs.cu
extern "C" void cuda_netCalcOutputs(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                                    float* results, unsigned input_size, unsigned output_size);
// calcInvertedMatrix.cu
extern "C" void
        cuda_netCalcInvMatrix(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                              float* results, unsigned input_size, unsigned output_size);

// calcOutputs.cu (utility that InvMatrix also uses)
unsigned getSharedMemorySize(BufferType inputType, unsigned input_size);

#endif /*CUDA_DEFINITIONS_H_*/
