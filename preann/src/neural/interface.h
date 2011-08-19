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
	BufferType bufferType;

public:
	Interface();
	Interface(FILE* stream);
	Interface(unsigned size, BufferType bufferType);
	Interface(Interface* toCopy);
	virtual ~Interface();

	void* getDataPointer();
	virtual unsigned getByteSize();

	unsigned getSize();
	BufferType getBufferType();
	float getElement(unsigned pos);
	void setElement(unsigned pos, float value);

	void copyFromFast(Interface* other);
	void copyFrom(Interface* other);
	void print();
	float compareTo(Interface* other);
	void random(float range);
    void reset();
	void save(FILE* stream);
	void load(FILE* stream);
	void transposeMatrix(unsigned width);
};

#endif /* INTERFACE_H_ */
