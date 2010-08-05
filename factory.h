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
	static Vector* newVector(ImplementationType implementationType);
	static Vector* newVector(unsigned size, VectorType vectorType, ImplementationType implementationType);
	static Layer* newLayer(ImplementationType implementationType);
	static Layer* newLayer(unsigned size, VectorType outputType, ImplementationType implementationType, FunctionType functionType);
};

#endif /* FACTORY_H_ */
