/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include "connection.h"

//TODO M intentar un solo archivo de configuracion para la factory (en vez de uno por target)
//TODO M evitar que se compile con nvcc cuando el target no lleva cuda

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
	static Connection* newThresholds(Vector* output, ImplementationType implementationType);
	static Connection* newConnection(FILE* stream, unsigned outputSize, ImplementationType implementationType);
};

#endif /* FACTORY_H_ */
