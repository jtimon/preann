#ifndef CUDABUFFER_H_
#define CUDABUFFER_H_

#include "neural/buffer.h"
#include "optimization/cuda_code.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaBuffer : virtual public Buffer
    {
    private:
        CudaBuffer()
        {
        }
        ;
    protected:
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
                    return (((tSize - 1) / BITS_PER_UNSIGNED) + 1)
                            * sizeof(unsigned);
            }
        }

        virtual void copyFromImpl(Interface *interface)
        {
            cuda_copyToDevice(data, interface->getDataPointer(),
                    interface->getByteSize());
        }

        virtual void copyToImpl(Interface *interface)
        {
            cuda_copyToHost(interface->getDataPointer(), data,
                    this->getByteSize());
        }
    public:

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA;
        }
        ;
        virtual BufferType getBufferType()
        {
            return bufferTypeTempl;
        }
        ;

        CudaBuffer(unsigned size)
        {
            this->tSize = size;

            unsigned byte_sz = getByteSize();
            data = cuda_malloc(byte_sz);

            cuda_setZero(data, byte_sz, bufferTypeTempl, CUDA_THREADS_PER_BLOCK);
        }
        //special constructor for bit coalescing buffers
        CudaBuffer(Interface* bitBuffer, unsigned block_size)
        {
            if (bitBuffer->getBufferType() != BT_BIT) {
                std::string error =
                        "The Buffer type must be BIT to use a BitBuffer CudaBuffer constructor.";
                throw error;
            }
            unsigned bitBufferSize = bitBuffer->getSize();
            unsigned maxWeighsPerBlock = BITS_PER_UNSIGNED * block_size;

            tSize = (bitBufferSize / maxWeighsPerBlock) * maxWeighsPerBlock;
            tSize += min(bitBufferSize % maxWeighsPerBlock, block_size)
                    * BITS_PER_UNSIGNED;

            Interface interfaceOrderedByBlockSize = Interface(tSize, BT_BIT);
            unsigned byteSize = interfaceOrderedByBlockSize.getByteSize();
            data = cuda_malloc(byteSize);

            unsigned bit = 0, thread = 0, block_offset = 0;
            for (unsigned i = 0; i < bitBufferSize; i++) {

                unsigned weighPos = (thread * BITS_PER_UNSIGNED) + bit
                        + block_offset;
                thread++;
                interfaceOrderedByBlockSize.setElement(weighPos,
                        bitBuffer->getElement(i));

                if (thread == block_size) {
                    thread = 0;
                    bit++;
                    if (bit == BITS_PER_UNSIGNED) {
                        bit = 0;
                        block_offset += (block_size * BITS_PER_UNSIGNED);
                    }
                }
            }
            cuda_copyToDevice(data,
                    interfaceOrderedByBlockSize.getDataPointer(), byteSize);
        }
        virtual ~CudaBuffer()
        {
            if (data) {
                cuda_free(data);
                data = NULL;
            }
        }

        virtual Buffer* clone()
        {
            Buffer* clone = new CudaBuffer(tSize);
            copyTo(clone);
            return clone;
        }

        virtual void activation(Buffer* resultsVect, FunctionType functionType)
        {
            float* results = (float*)resultsVect->getDataPointer();
            cuda_activation(data, tSize, bufferTypeTempl, results,
                    functionType, CUDA_THREADS_PER_BLOCK);
        }

    };

#endif /* CUDABUFFER_H_ */
