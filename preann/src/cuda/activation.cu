#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include "cuda.h"

__device__
float Func(float number, FunctionType functionType)
{
    switch (functionType) {

        case FT_BINARY_STEP:
            if (number > 0) {
                return 1;
            } else {
                return 0;
            }
        case FT_BIPOLAR_STEP:
            if (number > 0) {
                return 1;
            } else {
                return -1;
            }
        case SIGMOID:
            return 1.0f / (1.0f - exp(-number));
        case FT_BIPOLAR_SIGMOID:
            return -1.0f + (2.0f / (1.0f + exp(-number)));
        case FT_HYPERBOLIC_TANGENT:
            return tanh(number);
        case FT_IDENTITY:
        default:
            return number;
    }
}

__device__
unsigned device_min(unsigned a, unsigned b)
{
    return (a < b) ? a : b;
}

__global__
void ActivationFloatKernel(float* results, float* thresholds, float* output, unsigned output_sz,
                             FunctionType functionType)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_sz) {
        output[idx] = Func(results[idx] - thresholds[idx], functionType);
    }
}

__global__
void ActivationBitKernel(float* results, float* thresholds, unsigned* output, unsigned output_sz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned offset = idx * BITS_PER_UNSIGNED;

    if (output_sz > offset) {

        unsigned toRead = device_min(BITS_PER_UNSIGNED, output_sz - offset);
        unsigned threadOutput = 0;
        unsigned mask = 0x80000000;

        for (unsigned i = 0; i < toRead; i++) {
            unsigned pos = offset + i;
            if (results[pos] - thresholds[pos] > 0) {
                threadOutput |= mask;
            } else {
                threadOutput &= ~mask;
            }
            mask >>= 1;
        }
        output[idx] = threadOutput;
    }
}

extern "C" void cuda_activation(void* output, unsigned size, BufferType bufferType, float* results,
                                float* thresholds, FunctionType functionType, unsigned block_size)
{
    unsigned grid_size;

    switch (bufferType) {
        case BT_BYTE:
            {
                std::string error = "cuda_activation is not implemented for BufferType BYTE.";
                throw error;
            }
        case BT_FLOAT:
            {
                grid_size = ((size - 1) / block_size) + 1;
                ActivationFloatKernel<<< grid_size, block_size >>>(results, thresholds, (float*)output, size, functionType);
        }
        break;
        case BT_BIT:
        case BT_SIGN:
        {
            grid_size = ((size - 1) / (block_size * BITS_PER_UNSIGNED)) + 1;
            ActivationBitKernel<<< grid_size, block_size >>>(results, thresholds, (unsigned*)output, size);
        }
        break;
    }
    checkCUDAError("activation");
}
