/*
 * cuda2Connection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDA_REDUCCONNECTION0_H_
#define CUDA_REDUCCONNECTION0_H_

#include "cudaBaseConnection.h"

//special Buffer with a constructor for bit coalescing buffers when calling to cuda_crossoverOld
class CudaBitBuffer : virtual public CudaBuffer<BT_BIT, unsigned>
{
public:

    virtual ImplementationType getImplementationType()
    {
        return IT_CUDA_REDUC0;
    }

    CudaBitBuffer(Interface* bitBuffer, unsigned block_size)
    {
        if (bitBuffer->getBufferType() != BT_BIT) {
            std::string error = "The Buffer type must be BIT to use a BitBuffer CudaBuffer constructor.";
            throw error;
        }
        unsigned bitBufferSize = bitBuffer->getSize();
        unsigned maxWeighsPerBlock = BITS_PER_UNSIGNED * block_size;

        tSize = (bitBufferSize / maxWeighsPerBlock) * maxWeighsPerBlock;
        tSize += min(bitBufferSize % maxWeighsPerBlock, block_size) * BITS_PER_UNSIGNED;

        Interface interfaceOrderedByBlockSize = Interface(tSize, BT_BIT);

        unsigned bit = 0, thread = 0, block_offset = 0;
        for (unsigned i = 0; i < bitBufferSize; i++) {

            unsigned weighPos = (thread * BITS_PER_UNSIGNED) + bit + block_offset;
            thread++;
            interfaceOrderedByBlockSize.setElement(weighPos, bitBuffer->getElement(i));

            if (thread == block_size) {
                thread = 0;
                bit++;
                if (bit == BITS_PER_UNSIGNED) {
                    bit = 0;
                    block_offset += (block_size * BITS_PER_UNSIGNED);
                }
            }
        }
        unsigned byteSize = interfaceOrderedByBlockSize.getByteSize();
        data = cuda_malloc(byteSize);
        cuda_copyToDevice(data, interfaceOrderedByBlockSize.getDataPointer(), byteSize);
    }
    virtual ~CudaBitBuffer()
    {
    }
};


template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaReduction0Connection : virtual public Connection, public CudaBaseConnection<bufferTypeTempl,
            c_typeTempl>
    {
    protected:

        virtual void _calculateAndAddTo(Buffer* results)
        {
            float* resultsPtr = (float*) results->getDataPointer();
            // TODO TCC este método no funciona correctamente para BT_SIGN
            cuda_netCalcReduction0(tInput->getBufferType(), Cuda_Threads_Per_Block, tInput->getDataPointer(),
                                   this->getDataPointer(), resultsPtr, tInput->getSize(), results->getSize());
        }

        virtual void _crossover(Buffer* other, Interface* bitBuffer)
        {
            CudaBitBuffer cudaBitBuffer(bitBuffer, Cuda_Threads_Per_Block);

            cuda_crossoverOld(this->getDataPointer(), other->getDataPointer(),
                           (unsigned*) cudaBitBuffer.getDataPointer(), tSize, bufferTypeTempl,
                           Cuda_Threads_Per_Block);
        }

    public:
        CudaReduction0Connection(Buffer* input, unsigned outputSize) :
            CudaBaseConnection<bufferTypeTempl, c_typeTempl> (input, outputSize)
        {
        }

        virtual ~CudaReduction0Connection()
        {
        }

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA_REDUC0;
        }

    };

#endif /* CUDA_REDUCCONNECTION0_H_ */
