/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */

#ifndef FACTORY_H_
#define FACTORY_H_

//#include "cudaLayer.h"
//#include "xmmLayer.h"
#include "layer.h"

typedef enum {C, SSE2, CUDA, CUDA2} ImplementationType;

class Factory {
public:
	static Vector* newVector(ImplementationType implementationType, unsigned size, VectorType vectorType);
	static Layer* newLayer(ImplementationType implementationType, FunctionType functionType = IDENTITY, VectorType inputType = FLOAT, VectorType outputType = FLOAT);
};

#endif /* FACTORY_H_ */
