/*
 * configFactory.h
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#ifndef CONFIGFACTORY_H_
#define CONFIGFACTORY_H_

#include "template/cppConnection.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
Vector* func_newVector(unsigned size, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppVector<vectorTypeTempl, c_typeTempl>(size);
		case SSE2:
			{
			std::string error = "Implementation SSE2 is not allowed.";
			throw error;
			}
		case CUDA:
		case CUDA2:
		case CUDA_INV:
			{
			std::string error = "Implementation CUDA is not allowed.";
			throw error;
			}
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
Connection* func_newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case SSE2:
			{
			std::string error = "Implementation SSE2 is not allowed.";
			throw error;
			}
		case CUDA:
		case CUDA2:
		case CUDA_INV:
			{
			std::string error = "Implementation CUDA is not allowed.";
			throw error;
			}
	}
}

#endif /* CONFIGFACTORY_H_ */
