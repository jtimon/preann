/*
 * interface.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */

#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "util.h"

class Interface {

	unsigned size;
	void* data;
	VectorType vectorType;

public:
	Interface();
	Interface(unsigned size, VectorType vectorType);
	Interface(Interface* toCopy);
	virtual ~Interface();

	void* getDataPointer();
	virtual unsigned getByteSize();

	unsigned getSize();
	VectorType getVectorType();
	float getElement(unsigned pos);
	void setElement(unsigned pos, float value);

	void print();
	float compareTo(Interface* other);
	void random(float range);
	void save(FILE* stream);
	void load(FILE* stream);
	void transposeMatrix(unsigned width);
};

#endif /* INTERFACE_H_ */
