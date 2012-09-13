#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include "cuda.h"

__device__
unsigned device_min(unsigned a, unsigned b)
{
    return (a < b) ? a : b;
}

// CL_LAYER CALCULATION

__global__
void OutputsFloatKernel(float* inputs, float* weighs, float* results, unsigned input_size, unsigned output_size)
{
    extern __shared__ float sdata[];

    unsigned pos = threadIdx.x;
    while (pos < input_size) {

        sdata[pos] = inputs[pos];
        pos += blockDim.x;
    }
    __syncthreads();

    unsigned outputNeuron = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned weighsOffset = outputNeuron * input_size;
    float result = 0;

    if (outputNeuron < output_size) {

        for (unsigned i = 0; i < input_size; i++) {
            result += sdata[i] * weighs[weighsOffset + i];
        }
        results[outputNeuron] += result;
    }
}

template <BufferType inputType>
__global__
void OutputsBitKernel(unsigned* inputs, unsigned char* weighs, float* results, unsigned input_size, unsigned output_size)
{
    extern __shared__ unsigned shared_inputs[];

    unsigned tid = threadIdx.x;
    unsigned input_blocks_to_read = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
    unsigned readingLoops = ((input_blocks_to_read - 1) / blockDim.x) + 1;

    unsigned pos = tid;

    for (unsigned i=0; i < readingLoops; i++) {
        if (pos < input_blocks_to_read) {
            shared_inputs[pos] = inputs[pos];
        }
        pos += blockDim.x;
    }
    __syncthreads();

    unsigned outputNeuron = blockIdx.x*blockDim.x + threadIdx.x;
    if (outputNeuron < output_size) {

        float result = 0;
        unsigned weighsOffset = (outputNeuron * input_size);

        for (unsigned i=0; i < input_blocks_to_read; i++) {

            unsigned maxBits = device_min(BITS_PER_UNSIGNED, input_size - (i * BITS_PER_UNSIGNED));

            unsigned input_block = shared_inputs[i];
            unsigned mask = 0x80000000;
            for (unsigned j=0; j < maxBits; j++) {

                if (input_block & mask) {
                    result += weighs[weighsOffset] - 128;
                } else {
                    if (inputType == BT_SIGN) {
                        result += 128 - weighs[weighsOffset];
                    }
                }
                ++weighsOffset;
                mask >>= 1;
            }
        }
        results[outputNeuron] += result;
    }
}

unsigned getSharedMemorySize(BufferType inputType, unsigned input_size)
{
    unsigned size = 0;
    if (inputType == BT_BIT || inputType == BT_SIGN)
    {
        size = ( ( (input_size - 1) / BITS_PER_UNSIGNED ) + 1) * sizeof(unsigned);
    }
    else if (inputType == BT_FLOAT)
    {
        size = input_size * sizeof(float);
    }
    return (size < CUDA_MAX_SHARED_SIZE) ? size : CUDA_MAX_SHARED_SIZE;
};

extern "C" void cuda_netCalcOutputs(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                                    float* results, unsigned input_size, unsigned output_size)
{
    if (inputType == BT_BYTE) {
        std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
        throw error;
    }

    unsigned grid_size = ((output_size - 1) / block_size) + 1;
    unsigned shared_mem_size = getSharedMemorySize(inputType, input_size);

    if (shared_mem_size > CUDA_MAX_SHARED_SIZE) {
        string error;
        if (inputType == BT_FLOAT) {
            error = "The maximum float input size for cuda_netCalcOutputs is 4032.";
        } else {
            error = "The maximum bit/sign input size for cuda_netCalcOutputs is 129024.";
        }
        throw error;
    }

    if (inputType == BT_FLOAT) {
        OutputsFloatKernel<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtr, (float*)weighs, results, input_size, output_size);
    } else if (inputType == BT_BIT) {
        OutputsBitKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, (unsigned char*)weighs, results, input_size, output_size);
    } else {
        OutputsBitKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, (unsigned char*)weighs, results, input_size, output_size);
    }
}
