/*
 * vector.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "vector.h"

Vector::Vector()
{
	this->size = 0;
	data = NULL;
}

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

Interface* Vector::toInterface()
{
	Interface* toReturn = new Interface(this->size, this->vectorType);
	this->copyTo(toReturn);
	return toReturn;
}

void Vector::print()
{
	Interface* interface = toInterface();
	interface->print();
	delete(interface);
}

void Vector::save(FILE* stream)
{
	Interface* interface = toInterface();
	interface->save(stream);
	delete(interface);
}
