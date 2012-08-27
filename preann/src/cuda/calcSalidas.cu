#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include "cuda.h"

__device__
unsigned device_min(unsigned a, unsigned b)
{
    if (a < b)
        return a;
    return b;
}

// CL_LAYER CALCULATION

__global__
void SumFloatsConnectionsKernel(float* inputs, unsigned input_size, unsigned output_size, float* weighs,
                                float* results)
{
    extern __shared__ float sdata[];

    unsigned outputNeuron = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned weighsOffset = outputNeuron * input_size;
    float result = 0;

    unsigned pos = threadIdx.x;
    while (pos < input_size) {

        sdata[pos] = inputs[pos];
        pos += blockDim.x;
    }
    __syncthreads();

    if (outputNeuron < output_size) {

        //////////////////////////
        for (unsigned i = 0; i < input_size; i++) {
            result += sdata[i] * weighs[weighsOffset + i];
            //printf(" peso %f ", weighs[weighsOffset + i]);
        }
        /////TODO TR OTRA OPCION
        /*	if (blockDim.x <= input_size){
         unsigned pos = tid;
         while (pos < input_size){
         result += sdata[pos] * weighs[weighsOffset + pos];
         ++pos;
         }
         pos = 0;
         while (pos < tid){
         result += sdata[pos] * weighs[weighsOffset + pos];
         ++pos;
         }
         } else {
         unsigned pos = tid;
         while (pos < input_size){
         result += sdata[pos] * weighs[weighsOffset + pos];
         ++pos;
         }
         unsigned newMax = device_min(tid, input_size);
         pos = 0;
         while (pos < newMax){
         result += sdata[pos] * weighs[weighsOffset + pos];
         ++pos;
         }
         }*/
        /////////////
        results[outputNeuron] += result;
    }
}

template <BufferType inputType>
__global__
void SumBitsConnectionsKernel(unsigned* inputs, unsigned input_size, unsigned output_size, unsigned char* weighs, float* results)
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

            //TODO TCC check performance penalty (this is just for BT_SIGN)
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

extern "C" void cuda_inputCalculation(void* inputPtr, unsigned input_size, BufferType inputType,
                                      unsigned output_size, void* weighs, float* results, unsigned block_size)
{
    unsigned grid_size = ((output_size - 1) / block_size) + 1;
    unsigned shared_mem_size;

    if (inputType == BT_BYTE) {
        std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
        throw error;
    } else if (inputType == BT_FLOAT) {
        if (input_size > 4032) {
            string error = "The maximum float input size is 4032.";
            throw error;
        }
        shared_mem_size = input_size * sizeof(float);

        SumFloatsConnectionsKernel<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtr, input_size, output_size, (float*)weighs, results);
    } else {

        shared_mem_size =(((input_size - 1)/BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
        if (shared_mem_size > 16128) {
            //16128 * 8
            string error = "The maximum bit/sign input size is 129024.";
            throw error;
        }
        if (inputType == BT_BIT) {
            SumBitsConnectionsKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
        } else {
            SumBitsConnectionsKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
        }
    }
}
