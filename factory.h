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

typedef enum {C, SSE2, CUDA} ImplementationType;

class Factory {
public:
	Factory();
	virtual ~Factory();
	static Vector* newVector(ImplementationType implementationType, unsigned size, VectorType vectorType);
	static Layer* newLayer(ImplementationType implementationType);
	static Layer* newLayer(ImplementationType implementationType, VectorType inputType, VectorType outputType, FunctionType functionType);
};

#endif /* FACTORY_H_ */
