/*
 * configFactory.h
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#ifndef CONFIGFACTORY_H_
#define CONFIGFACTORY_H_

#ifdef CPP_IMPL
#include "cpp/cppConnection.h"
#endif
#ifdef SSE2_IMPL
#include "sse2/xmmConnection.h"
#endif
#ifdef CUDA_IMPL
#include "cuda/cudaReduction0Connection.h"
#include "cuda/cudaReductionConnection.h"
#include "cuda/cudaOutputsConnection.h"
#include "cuda/cudaInvertedConnection.h"
#endif

template<BufferType bufferTypeTempl, class c_typeTempl>
    Buffer* func_newBuffer(unsigned size, ImplementationType implementationType)
    {
        switch (implementationType) {
            case IT_C:
#ifdef CPP_IMPL
                return new CppBuffer<bufferTypeTempl, c_typeTempl>(size);
#else
                {
                    std::string error = "Implementation CPP is not allowed.";
                    throw error;
                }
#endif
            case IT_SSE2:
#ifdef SSE2_IMPL
                return new XmmBuffer<bufferTypeTempl, c_typeTempl>(size);
#else
                {
                    std::string error = "Implementation SSE2 is not allowed.";
                    throw error;
                }
#endif
            case IT_CUDA_OUT:
            case IT_CUDA_REDUC0:
            case IT_CUDA_REDUC:
            case IT_CUDA_INV:
#ifdef CUDA_IMPL
                return new CudaBuffer<bufferTypeTempl, c_typeTempl>(size);
#else
                {
                    std::string error = "Implementation CUDA is not allowed.";
                    throw error;
                }
#endif
            default:
                {
                    std::string error = "Unknown Implementation.";
                    throw error;
                }
        }
    }

template<BufferType bufferTypeTempl, class c_typeTempl>
    Connection* func_newConnection(Buffer* input, unsigned outputSize, ImplementationType implementationType)
    {
        switch (implementationType) {
            case IT_C:
#ifdef CPP_IMPL
                return new CppConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
#else
                {
                    std::string error = "Implementation CPP is not allowed.";
                    throw error;
                }
#endif
            case IT_SSE2:
#ifdef SSE2_IMPL
                return new XmmConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
#else
                {
                    std::string error = "Implementation SSE2 is not allowed.";
                    throw error;
                }
#endif
#ifdef CUDA_IMPL
            case IT_CUDA_OUT:
                return new CudaOutputsConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
            case IT_CUDA_REDUC0:
                return new CudaReduction0Connection<bufferTypeTempl, c_typeTempl>(input, outputSize);
            case IT_CUDA_REDUC:
                return new CudaReductionConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
            case IT_CUDA_INV:
                return new CudaInvertedConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
#else
                case IT_CUDA_OUT:
                case IT_CUDA_REDUC0:
                case IT_CUDA_REDUC:
                case IT_CUDA_INV:
                {
                    std::string error = "Implementation CUDA is not allowed.";
                    throw error;
                }
#endif
            default:
                {
                    std::string error = "Unknown Implementation.";
                    throw error;
                }
        }
    }

#endif /* CONFIGFACTORY_H_ */
