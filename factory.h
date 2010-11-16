/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include "connection.h"

class Factory {
protected:
	static VectorType weighForInput(VectorType inputType);
public:
	static void saveVector(Vector* vector, FILE* stream);
	static Vector* newVector(FILE* stream, ImplementationType implementationType);

	static Vector* newVector(Interface* interface, ImplementationType implementationType);
	static Vector* newVector(Vector* vector, ImplementationType implementationType);
	static Vector* newVector(unsigned size, VectorType vectorType, ImplementationType implementationType);
	static Connection* newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType);
	static Connection* newConnection(FILE* stream, unsigned outputSize, ImplementationType implementationType);
};

#endif /* FACTORY_H_ */
