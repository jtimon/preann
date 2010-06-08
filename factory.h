/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include "layer.h"
//TODO se requiere para el atributo static
#include "cudaLayer.h"

class Factory {
public:
	static Vector* newVector(unsigned size, VectorType vectorType, ImplementationType implementationType = SSE2, FunctionType functionType = IDENTITY);
	static Layer* newLayer(unsigned size, VectorType outputType, FunctionType functionType = IDENTITY, ImplementationType implementationType = SSE2);
};

#endif /* FACTORY_H_ */
