#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include "cuda.h"

__device__
unsigned device_min(unsigned a, unsigned b)
{
    return (a < b) ? a : b;
}

// GENETIC OPERATORS

// blockDim.x = BITS_PER_UNSIGNED : 32 threads that are going to process 32 * 32 = 1024 weighs
template <class type>
__global__
void crossoverSharedKernel(type* buffer1, type* buffer2, unsigned* bitBuffer, unsigned size)
{
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned bitBlocks =  ( (size - 1) / BITS_PER_UNSIGNED ) + 1;

    if (idx < bitBlocks) {
        sdata[threadIdx.x] = bitBuffer[ idx ];
    }
    __syncthreads();

    unsigned weighPos = (blockIdx.x * blockDim.x * BITS_PER_UNSIGNED) + threadIdx.x;
    for (int bit_block = 0; bit_block < blockDim.x; ++bit_block) {

        unsigned bit = sdata[ bit_block ];
        __syncthreads();

        unsigned mask = 0x80000000;
        mask >>= ( threadIdx.x );

        if (bit & mask) {
            type aux = buffer1[weighPos];
            buffer1[weighPos] = buffer2[weighPos];
            buffer2[weighPos] = aux;
        }
        weighPos += blockDim.x;
        __syncthreads();
    }
}

extern "C" void cuda_crossover(void* buffer1, void* buffer2, unsigned* bitBuffer, unsigned size,
                               BufferType bufferType)
{
    unsigned grid_size = ((size - 1) / (BITS_PER_UNSIGNED * BITS_PER_UNSIGNED)) + 1;

    switch (bufferType) {
        case BT_BYTE:
            crossoverSharedKernel<unsigned char><<< grid_size, BITS_PER_UNSIGNED >>>
                ((unsigned char*)buffer1, (unsigned char*)buffer2, bitBuffer, size);

            break;
        case BT_FLOAT:
            crossoverSharedKernel<float><<< grid_size, BITS_PER_UNSIGNED >>>
                ((float*)buffer1, (float*)buffer2, bitBuffer, size);
            break;
        case BT_BIT:
        case BT_SIGN:
            {
                std::string error = "cuda_crossover is not implemented for BufferType BIT nor SIGN.";
                throw error;
            }
    }
}

template <class type>
__global__
void CrossoverOldKernel(type* buffer1, type* buffer2, unsigned* bitBuffer, unsigned size)
{
    unsigned weighPos = (blockIdx.x * blockDim.x * BITS_PER_UNSIGNED) + threadIdx.x;
    unsigned maxPosForThisBlock = device_min ( (blockIdx.x + 1) * blockDim.x * BITS_PER_UNSIGNED, size);

    unsigned bitsForTheThread, mask;
    if (weighPos < maxPosForThisBlock) {
        bitsForTheThread = bitBuffer[(blockIdx.x * blockDim.x) + threadIdx.x];
        mask = 0x80000000;
    }
    __syncthreads();

    while (weighPos < maxPosForThisBlock) {
        if (mask & bitsForTheThread) {
            type aux = buffer1[weighPos];
            buffer1[weighPos] = buffer2[weighPos];
            buffer2[weighPos] = aux;
        }
        weighPos += blockDim.x;
        mask >>= 1;
    }
}

extern "C" void cuda_crossoverOld(void* buffer1, void* buffer2, unsigned* bitBuffer, unsigned size,
                               BufferType bufferType, unsigned block_size)
{
    unsigned grid_size = ((size - 1) / (block_size * BITS_PER_UNSIGNED)) + 1;

    switch (bufferType) {
        case BT_BYTE:
            CrossoverOldKernel<unsigned char><<< grid_size, block_size >>>
            ((unsigned char*)buffer1, (unsigned char*)buffer2, bitBuffer, size);

            break;
        case BT_FLOAT:
            CrossoverOldKernel<float><<< grid_size, block_size >>>
            ((float*)buffer1, (float*)buffer2, bitBuffer, size);
            break;
        case BT_BIT:
        case BT_SIGN:
            {
                std::string error = "cuda_crossover is not implemented for BufferType BIT nor SIGN.";
                throw error;
            }
    }
}

__global__
void resetFloatKernel(float* buffer, unsigned pos)
{
    if (threadIdx.x == 0) {
        buffer[pos] = 0;
    }
}

__global__
void resetByteKernel(unsigned char* buffer, unsigned pos)
{
    if (threadIdx.x == 0) {
        buffer[pos] = 128;
    }
}

__global__
void mutateFloatKernel(float* buffer, unsigned pos, float mutation)
{
    if (threadIdx.x == 0) {
        buffer[pos] += mutation;
    }
}

__global__
void mutateByteKernel(unsigned char* buffer, unsigned pos, int mutation)
{
    if (threadIdx.x == 0) {
        int result = mutation + buffer[pos];
        if (result <= 0) {
            buffer[pos] = 0;
        } else if (result >= 255) {
            buffer[pos] = 255;
        } else {
            buffer[pos] = (unsigned char) result;
        }
    }
}

extern "C" void cuda_mutateWeigh(void* buffer, unsigned pos, float mutation, BufferType bufferType)
{
    switch (bufferType) {
        case BT_BYTE:
            mutateByteKernel<<< 1, 8 >>>((unsigned char*)buffer, pos, (int)mutation);
            break;
        case BT_FLOAT:
            mutateFloatKernel<<< 1, 8 >>>((float*)buffer, pos, mutation);
            break;
        case BT_BIT:
        case BT_SIGN:
            {
                std::string error = "cuda_mutateWeigh is not implemented for BufferType BIT nor SIGN.";
                throw error;
            }
    }
}

extern "C" void cuda_resetWeigh(void* buffer, unsigned pos, BufferType bufferType)
{
    switch (bufferType) {
        case BT_BYTE:
            resetByteKernel<<< 1, 8 >>>((unsigned char*)buffer, pos);
            break;
        case BT_FLOAT:
            resetFloatKernel<<< 1, 8 >>>((float*)buffer, pos);
            break;
        case BT_BIT:
        case BT_SIGN:
            {
                std::string error = "cuda_resetWeigh is not implemented for BufferType BIT nor SIGN.";
                throw error;
            }
    }
}
