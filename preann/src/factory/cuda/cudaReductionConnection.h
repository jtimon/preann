/*
 * cuda2Connection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDA_REDUCCONNECTION_H_
#define CUDA_REDUCCONNECTION_H_

#include "cudaBaseConnection.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaReductionConnection : virtual public Connection, public CudaBaseConnection<bufferTypeTempl,
            c_typeTempl>
    {
    protected:
        virtual void _calculateAndAddTo(Buffer* results)
        {
            float* resultsPtr = (float*) results->getDataPointer();
            cuda_netCalcReduction(tInput->getBufferType(), Cuda_Threads_Per_Block, tInput->getDataPointer(),
                                  this->getDataPointer(), resultsPtr, tInput->getSize(), results->getSize());
        }

    public:
        CudaReductionConnection(Buffer* input, unsigned outputSize) :
            CudaBaseConnection<bufferTypeTempl, c_typeTempl> (input, outputSize)
        {
        }

        virtual ~CudaReductionConnection()
        {
        }

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA_REDUC;
        }

    };

#endif /* CUDA_REDUCCONNECTION_H_ */
