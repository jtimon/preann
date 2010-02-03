/*
 * vector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "generalDefinitions.h"
#include "xmmDefinitions.h"

class Vector {
protected:

	unsigned size;
	void* data;
	VectorType vectorType;

	virtual unsigned posToUnsignedPos(unsigned pos);
	virtual unsigned posToBitPos(unsigned pos);
public:
	Vector();
	Vector(unsigned size, VectorType vectorType);
	virtual ~Vector();
	void* getDataPointer();
	virtual unsigned getByteSize();
	unsigned getSize();
	VectorType getVectorType();

	virtual float getElement(unsigned pos);
	virtual void setElement(unsigned pos, float value);
	unsigned getWeighsSize();
	virtual void freeVector();

	void showVector();
};

#endif /* VECTOR_H_ */
