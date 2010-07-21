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

	Vector(){};
public:
	virtual ~Vector();

	virtual void copyFrom(Interface* interface) = 0;
	virtual void copyTo(Interface* interface) = 0;
	virtual void activation(float* results, FunctionType functionType) = 0;

	void* getDataPointer();
	unsigned getSize();
	VectorType getVectorType();
	FunctionType getFunctionType();

	void print();
};

#endif /* VECTOR_H_ */
