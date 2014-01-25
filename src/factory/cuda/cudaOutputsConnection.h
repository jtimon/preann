#ifndef CUDA_OUTPUTSCONNECTION_H_
#define CUDA_OUTPUTSCONNECTION_H_

#include "cudaBaseConnection.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CudaOutputsConnection : virtual public Connection, public CudaBaseConnection<bufferTypeTempl,
            c_typeTempl>
    {
    protected:
        virtual void _calculateAndAddTo(Buffer* results)
        {
            float* resultsPtr = (float*) results->getDataPointer();
            cuda_netCalcOutputs(tInput->getBufferType(), Cuda_Threads_Per_Block, tInput->getDataPointer(),
                                this->getDataPointer(), resultsPtr, tInput->getSize(), results->getSize());
        }

    public:
        CudaOutputsConnection(Buffer* input, unsigned outputSize) :
            CudaBaseConnection<bufferTypeTempl, c_typeTempl> (input, outputSize)
        {
        }

        virtual ~CudaOutputsConnection()
        {
        }

        virtual ImplementationType getImplementationType()
        {
            return IT_CUDA_OUT;
        }

    };

#endif /* CUDA_OUTPUTSCONNECTION_H_ */
