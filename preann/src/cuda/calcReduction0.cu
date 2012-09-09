#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include "cuda.h"

__device__
unsigned device_min(unsigned a, unsigned b)
{
    return (a < b) ? a : b;
}

template <BufferType inputType>
__global__
void Reduction0Kernel(void* inputPtr, void* weighs, float* results, unsigned input_size)
{
    extern __shared__ float sdata[];

    unsigned weighsOffset = (blockIdx.x * input_size);

    float result = 0;
    unsigned i = threadIdx.x;

    if (inputType == BT_FLOAT) {
        while (i < input_size) {
            result += ((float*)inputPtr)[i] * ((float*)weighs)[weighsOffset + i];
            i += blockDim.x;
        }
    } else {
        weighsOffset += threadIdx.x * BITS_PER_UNSIGNED;

        unsigned input_blocks_to_read = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
        while (i < input_blocks_to_read) {

            unsigned maxBits = device_min(BITS_PER_UNSIGNED, input_size - (i * BITS_PER_UNSIGNED));

            unsigned mask = 0x80000000;
            unsigned currentInput = ((unsigned*)inputPtr)[i];

            for (unsigned j=0; j < maxBits; j++) {

                if (currentInput & mask) {
                    result += ((unsigned char*)weighs)[weighsOffset + j] - 128;
                } else {
                    if (inputType == BT_SIGN) {
                        result -= ((unsigned char*)weighs)[weighsOffset + j] - 128;
                    }
                }
                mask >>= 1;
            }
            i += blockDim.x;
            weighsOffset += blockDim.x * BITS_PER_UNSIGNED;
        }
    }
//    __syncthreads();

    unsigned tid = threadIdx.x;
    sdata[tid] = result;
//    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
//        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        results[blockIdx.x] += sdata[0];
    }
}

extern "C" void cuda_netCalcReduction0(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                                       float* results, unsigned input_size, unsigned output_size)
{
    unsigned grid_size = output_size;
    unsigned shared_mem_size = block_size * sizeof(float);

    switch(inputType) {
        case BT_BYTE:
        {
            std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
            throw error;
        }
        case BT_FLOAT:
            Reduction0Kernel<BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, weighs, results, input_size);
            break;
        case BT_BIT:
            Reduction0Kernel<  BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, weighs, results, input_size);
            break;
        case BT_SIGN:
            Reduction0Kernel< BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, weighs, results, input_size);
            break;

    }
    checkCUDAError("cuda_inputCalculation2");
}

