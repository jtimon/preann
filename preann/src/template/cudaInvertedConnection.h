#ifndef CUDAINVERTEDCONNECTION_H_
#define CUDAINVERTEDCONNECTION_H_

#include "neural/connection.h"
#include "cudaBuffer.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaInvertedConnection : public virtual FullConnection,
            public CudaBuffer<bufferTypeTempl, c_typeTempl>
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

        unsigned invertPos(unsigned pos)
        {
            //TODO z simplificar cuentas
            unsigned outputPos = pos / tInput->getSize();
            unsigned inputPos = (pos % tInput->getSize());
            unsigned outputSize = tSize / tInput->getSize();
            return outputPos + (inputPos * outputSize);
        }

        virtual void _mutateWeigh(unsigned pos, float mutation)
        {
            cuda_mutateWeigh(data, invertPos(pos), mutation, bufferTypeTempl);
        }

        virtual void _resetWeigh(unsigned pos)
        {
            cuda_resetWeigh(data, invertPos(pos), bufferTypeTempl);
        }

        virtual void _crossover(Buffer* other, Interface* bitBuffer)
        {
            Interface invertedBitBuffer = Interface(bitBuffer);
            invertedBitBuffer.transposeMatrix(tInput->getSize());

            CudaBuffer<bufferTypeTempl, c_typeTempl> cudaBitBuffer(
                    &invertedBitBuffer, Cuda_Threads_Per_Block);

            cuda_crossover(this->getDataPointer(), other->getDataPointer(),
                    (unsigned*)cudaBitBuffer.getDataPointer(), tSize,
                    bufferTypeTempl, Cuda_Threads_Per_Block);
        }
    public:
        CudaInvertedConnection(Buffer* input, unsigned outputSize) :
            CudaBuffer<bufferTypeTempl, c_typeTempl> (input->getSize()
                    * outputSize)
        {
            tInput = input;
        }

        virtual ~CudaInvertedConnection()
        {
        }
        ;

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA_INV;
        }
        ;

        virtual void _calculateAndAddTo(Buffer* results)
        {
            void* inputWeighs = this->getDataPointer();
            float* resultsPtr = (float*)results->getDataPointer();

            cuda_inputCalculationInvertedMatrix(tInput->getDataPointer(),
                    tInput->getSize(), tInput->getBufferType(),
                    results->getSize(), inputWeighs, resultsPtr,
                    Cuda_Threads_Per_Block);
        }

    };

#endif /* CUDAINVERTEDCONNECTION_H_ */
