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

template <VectorType vectorTypeTempl, class c_typeTempl>
Vector* func_newVector(unsigned size, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
#ifdef CPP_IMPL
			return new CppVector<vectorTypeTempl, c_typeTempl>(size);
#else
			{
			std::string error = "Implementation CPP is not allowed.";
			throw error;
			}
#endif
		case SSE2:
#ifdef SSE2_IMPL
			return new XmmVector<vectorTypeTempl, c_typeTempl>(size);
#else
			{
			std::string error = "Implementation SSE2 is not allowed.";
			throw error;
			}
#endif
		case CUDA:
		case CUDA2:
		case CUDA_INV:
#ifdef CUDA_IMPL
			return new CudaVector<vectorTypeTempl, c_typeTempl>(size);
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

template <VectorType vectorTypeTempl, class c_typeTempl>
Connection* func_newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
#ifdef CPP_IMPL
			return new CppConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
#else
			{
			std::string error = "Implementation CPP is not allowed.";
			throw error;
			}
#endif
		case SSE2:
#ifdef SSE2_IMPL
			return new XmmConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
#else
			{
			std::string error = "Implementation SSE2 is not allowed.";
			throw error;
			}
#endif
#ifdef CUDA_IMPL
		case CUDA:
			return new CudaConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA2:
			return new Cuda2Connection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA_INV:
			return new CudaInvertedConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
#else
		case CUDA:
		case CUDA2:
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
