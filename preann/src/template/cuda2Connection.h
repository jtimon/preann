/*
 * cuda2Connection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDA_REDUCCONNECTION_H_
#define CUDA_REDUCCONNECTION_H_

#include "cudaConnection.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class Cuda2Connection : virtual public FullConnection,
            public CudaConnection<bufferTypeTempl, c_typeTempl>
    {
    public:
        Cuda2Connection(Buffer* input, unsigned outputSize) :
            CudaConnection<bufferTypeTempl, c_typeTempl> (input, outputSize)
        {
        }

        virtual ~Cuda2Connection()
        {
        }
        ;

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA_REDUC;
        }
        ;

        virtual void calculateAndAddTo(Buffer* results)
        {
            void* inputWeighs = this->getDataPointer();
            float* resultsPtr = (float*)results->getDataPointer();
            // TODO TCC este método no funciona correctamente para BT_SIGN
            cuda_inputCalculationReduction(tInput->getDataPointer(),
                    tInput->getSize(), tInput->getBufferType(),
                    results->getSize(), inputWeighs, resultsPtr,
                    Cuda_Threads_Per_Block);
        }
    };

#endif /* CUDA_REDUCCONNECTION_H_ */
