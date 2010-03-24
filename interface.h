/*
 * interface.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */

#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "generalDefinitions.h"

class Interface {

	unsigned size;
	void* data;
	VectorType vectorType;

public:
	Interface(unsigned size, VectorType vectorType);
	virtual ~Interface();

	void* getDataPointer();
	virtual unsigned getByteSize();

	unsigned getSize();
	VectorType getVectorType();
	virtual float getElement(unsigned pos);
	virtual void setElement(unsigned pos, float value);

	void print();
	float compareTo(Interface* other);
};

#endif /* INTERFACE_H_ */
