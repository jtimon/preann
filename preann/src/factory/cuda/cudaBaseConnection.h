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
    class CudaBaseConnection : public virtual Connection, public CudaBuffer<bufferTypeTempl, c_typeTempl>
    {
    protected:
        virtual void _activation(Buffer* output, FunctionType functionType)
        {
            void* outputData = output->getDataPointer();
            float* results = (float*) tInput->getDataPointer();
            float* thresholds = (float*) data;

            cuda_activation(outputData, tSize, output->getBufferType(), results, thresholds, functionType,
                            Cuda_Threads_Per_Block);
        }

        virtual void _crossover(Buffer* other, Interface* bitBuffer)
        {
            Buffer* cudaBitBuffer = Factory::newBuffer(bitBuffer, getImplementationType());

            cuda_crossover(this->getDataPointer(), other->getDataPointer(),
                           (unsigned*) cudaBitBuffer->getDataPointer(), tSize, bufferTypeTempl);

            delete(cudaBitBuffer);
        }

        virtual void _mutateWeigh(unsigned pos, float mutation)
        {
            cuda_mutateWeigh(data, pos, mutation, bufferTypeTempl);
        }

        virtual void _resetWeigh(unsigned pos)
        {
            cuda_resetWeigh(data, pos, bufferTypeTempl);
        }
    public:
        CudaBaseConnection(Buffer* input, unsigned outputSize) :
            CudaBuffer<bufferTypeTempl, c_typeTempl> (input->getSize() * outputSize)
        {
            tInput = input;
        }

        virtual ~CudaBaseConnection()
        {
        }

    };

#endif /* CUDACONNECTION_H_ */
