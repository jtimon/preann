#ifndef CUDABUFFER_H_
#define CUDABUFFER_H_

#include "neural/buffer.h"
#include "cuda/cuda.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaBuffer : virtual public Buffer
    {
    protected:
        CudaBuffer()
        {
        }
        unsigned getByteSize()
        {
            switch (bufferTypeTempl) {
                case BT_BYTE:
                    return tSize;
                    break;
                case BT_FLOAT:
                    return tSize * sizeof(float);
                case BT_BIT:
                case BT_SIGN:
                    return (((tSize - 1) / BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
            }
        }

        virtual void _copyFrom(Interface *interface)
        {
            cuda_copyToDevice(data, interface->getDataPointer(), interface->getByteSize());
        }

        virtual void _copyTo(Interface *interface)
        {
            cuda_copyToHost(interface->getDataPointer(), data, this->getByteSize());
        }
    public:

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA_OUT;
        }

        virtual BufferType getBufferType()
        {
            return bufferTypeTempl;
        }

        CudaBuffer(unsigned size)
        {
            Util::check(bufferTypeTempl == BT_FLOAT_SMALL, "BufferType FLOAT_SMALL is not allowed for CudaBuffer");
            this->tSize = size;

            unsigned byte_sz = getByteSize();
            data = cuda_malloc(byte_sz);

            reset();
        }

        //special constructor for bit coalescing buffers
        CudaBuffer(Interface* bitBuffer, unsigned block_size)
        {
            if (bitBuffer->getBufferType() != BT_BIT) {
                std::string error = "The Buffer type must be BIT to use a BitBuffer CudaBuffer constructor.";
                throw error;
            }
            unsigned bitBufferSize = bitBuffer->getSize();
            unsigned maxWeighsPerBlock = BITS_PER_UNSIGNED * block_size;

            tSize = (bitBufferSize / maxWeighsPerBlock) * maxWeighsPerBlock;
            tSize += min(bitBufferSize % maxWeighsPerBlock, block_size) * BITS_PER_UNSIGNED;

            Interface interfaceOrderedByBlockSize(tSize, BT_BIT);

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
        virtual ~CudaBuffer()
        {
            if (data) {
                cuda_free(data);
                data = NULL;
            }
        }

        virtual void reset()
        {
            unsigned byte_sz = getByteSize();

            cuda_setZero(data, byte_sz, bufferTypeTempl, Cuda_Threads_Per_Block);
        }
    };

#endif /* CUDABUFFER_H_ */
