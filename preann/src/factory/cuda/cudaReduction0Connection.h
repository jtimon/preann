/*
 * cuda2Connection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDA_REDUCCONNECTION0_H_
#define CUDA_REDUCCONNECTION0_H_

#include "cudaBaseConnection.h"

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
