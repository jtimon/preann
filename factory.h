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
#include "cudaLayer2.h"

typedef enum {C, SSE2, CUDA} ImplementationType;

class Factory {
public:
	static Vector* newVector(ImplementationType implementationType, unsigned size, VectorType vectorType);
	static Layer* newLayer(ImplementationType implementationType, FunctionType functionType = IDENTITY, VectorType inputType = FLOAT, VectorType outputType = FLOAT);
};

#endif /* FACTORY_H_ */
