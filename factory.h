/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include "layer.h"

class Factory {
public:
	static Vector* newVector(unsigned size, VectorType vectorType, ImplementationType implementationType = SSE2, FunctionType functionType = IDENTITY);
	static Layer* newLayer(ImplementationType implementationType);
};

#endif /* FACTORY_H_ */
