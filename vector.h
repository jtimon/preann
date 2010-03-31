/*
 * vector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "sse2_code.h"
#include "interface.h"

class Vector {
protected:

	unsigned size;
	void* data;
	VectorType vectorType;

	Vector() {};
	virtual unsigned getByteSize();
public:
	Vector(unsigned size, VectorType vectorType);
	virtual ~Vector();

	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(float* results, FunctionType functionType);

	void* getDataPointer();
	unsigned getSize();
	VectorType getVectorType();

	void print();
	//TODO eliminar o cambiar por weighsOffset (para poder aceptar varios tipos de entrada)
	unsigned getWeighsSize();
};

#endif /* VECTOR_H_ */
