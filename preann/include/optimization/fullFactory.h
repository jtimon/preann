/*
 * configFactory.h
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#ifndef CONFIGFACTORY_H_
#define CONFIGFACTORY_H_

#include "cppConnection.h"
#include "xmmConnection.h"
#include "cuda2Connection.h"
#include "cudaInvertedConnection.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
Vector* func_newVector(unsigned size, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppVector<vectorTypeTempl, c_typeTempl>(size);
		case SSE2:
			return new XmmVector<vectorTypeTempl, c_typeTempl>(size);
		case CUDA:
		case CUDA2:
		case CUDA_INV:
			return new CudaVector<vectorTypeTempl, c_typeTempl>(size);
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
Connection* func_newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case SSE2:
			return new XmmConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA:
			return new CudaConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA2:
			return new Cuda2Connection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA_INV:
			return new CudaInvertedConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
	}
}

#endif /* CONFIGFACTORY_H_ */
