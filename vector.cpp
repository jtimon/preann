/*
 * vector.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "vector.h"

Vector::~Vector()
{
}

void* Vector::getDataPointer()
{
	return data;
}

unsigned Vector::getSize()
{
	return size;
}

VectorType Vector::getVectorType()
{
	return vectorType;
}

Interface* Vector::createInterface()
{
	Interface* toReturn = new Interface(this->size, this->vectorType);
	this->copyTo(toReturn);
	return toReturn;
}

void Vector::print()
{
	Interface* interface = new Interface(size, vectorType);
	copyTo(interface);
	interface->print();
	delete(interface);
}


