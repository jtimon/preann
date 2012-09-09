#ifndef CUDAINVERTEDCONNECTION_H_
#define CUDAINVERTEDCONNECTION_H_

#include "neural/connection.h"
#include "cudaBaseConnection.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaInvertedConnection : public virtual Connection, public CudaBaseConnection<bufferTypeTempl,
            c_typeTempl>
    {
    protected:
        //redefined from CudaBuffer
        virtual void _copyFrom(Interface* interface)
        {
            interface->transposeMatrix(tInput->getSize());
            CudaBuffer<bufferTypeTempl, c_typeTempl>::_copyFrom(interface);
        }

        virtual void _copyTo(Interface* interface)
        {
            CudaBuffer<bufferTypeTempl, c_typeTempl>::_copyTo(interface);
            interface->transposeMatrix(tSize / tInput->getSize());
        }

        virtual void _calculateAndAddTo(Buffer* results)
        {
            float* resultsPtr = (float*) results->getDataPointer();

            cuda_netCalcInvMatrix(tInput->getBufferType(), Cuda_Threads_Per_Block, tInput->getDataPointer(),
                                  this->getDataPointer(), resultsPtr, tInput->getSize(), results->getSize());
        }

        unsigned invertPos(unsigned pos)
        {
            //TODO z simplificar cuentas
            unsigned outputPos = pos / tInput->getSize();
            unsigned inputPos = (pos % tInput->getSize());
            unsigned outputSize = tSize / tInput->getSize();
            return outputPos + (inputPos * outputSize);
        }

        virtual void _crossover(Buffer* other, Interface* bitBuffer)
        {
            Interface invertedBitBuffer = Interface(bitBuffer);
            invertedBitBuffer.transposeMatrix(tInput->getSize());

            CudaBuffer<bufferTypeTempl, c_typeTempl>
                    cudaBitBuffer(&invertedBitBuffer, Cuda_Threads_Per_Block);

            cuda_crossover(this->getDataPointer(), other->getDataPointer(),
                           (unsigned*) cudaBitBuffer.getDataPointer(), tSize, bufferTypeTempl,
                           Cuda_Threads_Per_Block);
        }

        virtual void _mutateWeigh(unsigned pos, float mutation)
        {
            cuda_mutateWeigh(data, invertPos(pos), mutation, bufferTypeTempl);
        }

        virtual void _resetWeigh(unsigned pos)
        {
            cuda_resetWeigh(data, invertPos(pos), bufferTypeTempl);
        }

    public:
        CudaInvertedConnection(Buffer* input, unsigned outputSize) :
            CudaBaseConnection<bufferTypeTempl, c_typeTempl> (input, outputSize)
        {
        }

        virtual ~CudaInvertedConnection()
        {
        }

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA_INV;
        }
    };

#endif /* CUDAINVERTEDCONNECTION_H_ */
