/*
 * cudaConnection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDACONNECTION_H_
#define CUDACONNECTION_H_

#include "neural/connection.h"
#include "cudaBuffer.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaConnection : public virtual Connection, public CudaBuffer<bufferTypeTempl, c_typeTempl>
    {
    protected:
        virtual void _mutateWeigh(unsigned pos, float mutation)
        {
            cuda_mutateWeigh(data, pos, mutation, bufferTypeTempl);
        }

        virtual void _resetWeigh(unsigned pos)
        {
            cuda_resetWeigh(data, pos, bufferTypeTempl);
        }

        virtual void _crossover(Buffer* other, Interface* bitBuffer)
        {
            CudaBuffer<bufferTypeTempl, c_typeTempl> cudaBitBuffer(bitBuffer, Cuda_Threads_Per_Block);

            cuda_crossover(this->getDataPointer(), other->getDataPointer(),
                           (unsigned*) cudaBitBuffer.getDataPointer(), tSize, bufferTypeTempl,
                           Cuda_Threads_Per_Block);
        }
    public:
        CudaConnection(Buffer* input, unsigned outputSize)
                : CudaBuffer<bufferTypeTempl, c_typeTempl>(input->getSize() * outputSize)
        {
            tInput = input;
        }

        virtual ~CudaConnection()
        {
        }
        ;

        virtual void _calculateAndAddTo(Buffer* results)
        {
            void* inputWeighs = this->getDataPointer();
            float* resultsPtr = (float*) results->getDataPointer();
            // TODO TCC este método no funciona correctamente para BT_SIGN
            cuda_inputCalculation(tInput->getDataPointer(), tInput->getSize(), tInput->getBufferType(),
                                  results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
        }

    };

#endif /* CUDACONNECTION_H_ */
