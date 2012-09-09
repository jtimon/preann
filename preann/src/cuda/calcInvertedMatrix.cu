#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include "cuda.h"

__device__
unsigned device_min(unsigned a, unsigned b)
{
    return (a < b) ? a : b;
}

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
            // TODO se puede poner esto aqui?
//            __syncthreads();
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

            //TODO TCC check performance penalty (this is just for BT_SIGN)
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
    unsigned grid_size = ((output_size - 1) / block_size) + 1;
    unsigned shared_mem_size;

    if (inputType == BT_BYTE) {
        std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
        throw error;
    } else if (inputType == BT_FLOAT) {

        float* flInputPtr = (float*)inputPtr;
        float* flWeighs = (float*)weighs;

        while (input_size > CUDA_MAX_SHARED_FLOATS) {

            shared_mem_size = CUDA_MAX_SHARED_FLOATS * sizeof(float);
            InvertedFloatKernel<<< grid_size, block_size, shared_mem_size >>>(flInputPtr, flWeighs, results, CUDA_MAX_SHARED_FLOATS, output_size);
            flInputPtr += CUDA_MAX_SHARED_FLOATS;
            flWeighs += (CUDA_MAX_SHARED_FLOATS * output_size);
            input_size -= CUDA_MAX_SHARED_FLOATS;
        }
        //TODO esto se puede meter dentro del bucle usando CUDA_MAX_SHARED_SIZE
        shared_mem_size = input_size * sizeof(float);
        InvertedFloatKernel<<< grid_size, block_size, shared_mem_size >>>(flInputPtr, flWeighs, results, input_size, output_size);
    } else {
        //TODO TCC esta parte no funciona bien
        while (input_size > CUDA_MAX_SHARED_BITS) {

// TODO uniformar con lo de arriba, que esto está super sucio
            shared_mem_size = CUDA_MAX_SHARED_FLOATS * sizeof(unsigned);
            // TODO TCC probar sin emulación
            if (inputType == BT_BIT) {
                InvertedBitKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, (unsigned char*)weighs, results, CUDA_MAX_SHARED_BITS, output_size);
            } else {
                InvertedBitKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, (unsigned char*)weighs, results, CUDA_MAX_SHARED_BITS, output_size);
            }
            inputPtr = (void*)((float*)inputPtr + CUDA_MAX_SHARED_FLOATS);
            weighs = (void*)((float*)weighs + (CUDA_MAX_SHARED_BITS * output_size));
            input_size -= CUDA_MAX_SHARED_BITS;
        }
        shared_mem_size =(((input_size - 1)/BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
        // TODO TCC probar sin emulación
        if (inputType == BT_BIT) {
            InvertedBitKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, (unsigned char*)weighs, results, input_size, output_size);
        } else {
            InvertedBitKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, (unsigned char*)weighs, results, input_size, output_size);
        }
    }
}
