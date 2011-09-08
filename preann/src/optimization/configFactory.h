/*
 * configFactory.h
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#ifndef CONFIGFACTORY_H_
#define CONFIGFACTORY_H_

#ifdef CPP_IMPL
	#include "cppConnection.h"
#endif
#ifdef SSE2_IMPL
	#include "xmmConnection.h"
#endif
#ifdef CUDA_IMPL
	#include "cuda2Connection.h"
	#include "cudaInvertedConnection.h"
#endif

template <BufferType bufferTypeTempl, class c_typeTempl>
Buffer* func_newBuffer(unsigned size, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
#ifdef CPP_IMPL
			return new CppBuffer<bufferTypeTempl, c_typeTempl>(size);
#else
			{
			std::string error = "Implementation CPP is not allowed.";
			throw error;
			}
#endif
		case SSE2:
#ifdef SSE2_IMPL
			return new XmmBuffer<bufferTypeTempl, c_typeTempl>(size);
#else
			{
			std::string error = "Implementation SSE2 is not allowed.";
			throw error;
			}
#endif
		case CUDA:
		case CUDA_REDUC:
		case CUDA_INV:
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

template <BufferType bufferTypeTempl, class c_typeTempl>
Connection* func_newConnection(Buffer* input, unsigned outputSize, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
#ifdef CPP_IMPL
			return new CppConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
#else
			{
			std::string error = "Implementation CPP is not allowed.";
			throw error;
			}
#endif
		case SSE2:
#ifdef SSE2_IMPL
			return new XmmConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
#else
			{
			std::string error = "Implementation SSE2 is not allowed.";
			throw error;
			}
#endif
#ifdef CUDA_IMPL
		case CUDA:
			return new CudaConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA_REDUC:
			return new Cuda2Connection<bufferTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA_INV:
			return new CudaInvertedConnection<bufferTypeTempl, c_typeTempl>(input, outputSize);
#else
		case CUDA:
		case CUDA_REDUC:
		case CUDA_INV:
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
