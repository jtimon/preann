#include "cudaCommon.h"

#define CUDA_BYTES_PARAMS_K_INVERTED (3 * size_of(void*) + 2 * size_of(unsigned))

__global__
void InvertedFloatKernel(float* inputs, float* weighs, float* results, unsigned input_size, unsigned output_size)
{
    extern __shared__ float sdata[];

    unsigned input_pos = threadIdx.x;
    while (input_pos < input_size) {

        sdata[input_pos] = inputs[input_pos];
        input_pos += blockDim.x;
    }
    __syncthreads();

    unsigned output_pos = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0;

    if (output_pos < output_size) {

        for (unsigned i = 0; i < input_size; i++) {
            result += sdata[i] * weighs[output_pos + (i * output_size)];
        }
        results[output_pos] += result;
    }
}

template <BufferType inputType>
__global__
void InvertedBitKernel(unsigned* inputs, unsigned char* weighs, float* results, unsigned input_size, unsigned output_size)
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

        for (unsigned i=0; i < input_blocks_to_read; i++) {

            unsigned maxBits = device_min(BITS_PER_UNSIGNED, input_size - (i * BITS_PER_UNSIGNED));

            unsigned weighsOffset = (i * BITS_PER_UNSIGNED * output_size) + outputNeuron;
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
                weighsOffset += output_size;
                mask >>= 1;
            }
        }
        results[outputNeuron] += result;
    }
}

extern "C" void cuda_netCalcInvMatrix(BufferType inputType, unsigned block_size, void* inputPtr, void* weighs,
                                      float* results, unsigned input_size, unsigned output_size)
{
    if (inputType == BT_BYTE) {
        std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
        throw error;
    }

    unsigned grid_size = ((output_size - 1) / block_size) + 1;
    unsigned char* inputPtrAux = (unsigned char*)inputPtr;
    unsigned inputs_to_process;
    unsigned shared_mem_size = getSharedMemorySize(inputType, input_size);

    while (shared_mem_size > 0) {

        if (inputType == BT_FLOAT) {
            inputs_to_process = shared_mem_size / 4;
            InvertedFloatKernel<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtrAux, (float*)weighs, results, inputs_to_process, output_size);
            weighs = (float*)weighs + (inputs_to_process * output_size);
        } else {
            inputs_to_process = (shared_mem_size / 4) * BITS_PER_UNSIGNED;
            if (inputType == BT_BIT) {
                InvertedBitKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtrAux, (unsigned char*)weighs, results, inputs_to_process, output_size);
            } else {
                InvertedBitKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtrAux, (unsigned char*)weighs, results, inputs_to_process, output_size);
            }
            weighs = (unsigned char*)weighs + (inputs_to_process * output_size);
        }

        inputPtrAux += shared_mem_size;
        input_size -= inputs_to_process;
        shared_mem_size = getSharedMemorySize(inputType, input_size);
    }
}
