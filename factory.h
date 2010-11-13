/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include "vector.h"

class Factory {
public:
	static void saveVector(Vector* vector, FILE* stream);
	static Vector* newVector(FILE* stream, ImplementationType implementationType);

	static Vector* newVector(Interface* interface, ImplementationType implementationType);
	static Vector* newVector(Vector* vector, ImplementationType implementationType);
	static Vector* newVector(unsigned size, VectorType vectorType, ImplementationType implementationType);
};

#endif /* FACTORY_H_ */
