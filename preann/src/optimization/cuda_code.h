#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "common/enumerations.h"

#define CUDA_THREADS_PER_BLOCK 64

#define CUDA_MAX_SHARED_FLOATS (4032)
#define CUDA_MAX_SHARED_BITS (CUDA_MAX_SHARED_FLOATS * CUDA_MAX_SHARED_FLOATS)

static unsigned Cuda_Threads_Per_Block = 512;

extern "C" void* cuda_malloc(unsigned byteSize);
extern "C" void cuda_free(void* d_ptr);
extern "C" void cuda_copyToDevice(void* d_dest, void* h_src, unsigned count);
extern "C" void cuda_copyToHost(void* h_dest, void* d_src, unsigned count);
extern "C" void cuda_setZero(void* data, unsigned byteSize,
        BufferType bufferType, unsigned block_size);

extern "C" void cuda_activation(void* data, unsigned size,
        BufferType bufferType, float* results, FunctionType functionType,
        unsigned block_size);

extern "C" void cuda_inputCalculation(void* inputPtr, unsigned input_size,
        BufferType inputType, unsigned output_size, void* weighs,
        float* results, unsigned block_size);
extern "C" void cuda_inputCalculationReduction(void* inputPtr,
        unsigned input_size, BufferType inputType, unsigned output_size,
        void* weighs, float* results, unsigned block_size);
extern "C" void cuda_inputCalculationInvertedMatrix(void* inputPtr,
        unsigned input_size, BufferType inputType, unsigned output_size,
        void* weighs, float* results, unsigned block_size);

extern "C" void cuda_mutateWeigh(void* buffer, unsigned pos, float mutation,
        BufferType bufferType);
extern "C" void cuda_resetWeigh(void* buffer, unsigned pos, BufferType bufferType);
extern "C" void cuda_crossover(void* buffer1, void* buffer2,
        unsigned* bitBuffer, unsigned size, BufferType bufferType,
        unsigned block_size);

#endif /*CUDA_DEFINITIONS_H_*/
