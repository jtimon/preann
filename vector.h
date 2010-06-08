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
	FunctionType functionType;

	Vector() {};
	virtual unsigned getByteSize();
public:
	Vector(unsigned size, VectorType vectorType, FunctionType functionType);
	virtual ~Vector();

	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(float* results);

	void* getDataPointer();
	unsigned getSize();
	VectorType getVectorType();

	void print();
    FunctionType getFunctionType();

};

#endif /* VECTOR_H_ */
