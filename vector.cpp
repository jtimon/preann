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

void Vector::copyFromVector(Vector* vector)
{
	Interface* interface = vector->toInterface();
	this->copyFrom(interface);
	delete(interface);
}

void Vector::copyToVector(Vector* vector)
{
	Interface* interface = this->toInterface();
	vector->copyFrom(interface);
	delete(interface);
}

void Vector::print()
{
	Interface* interface = toInterface();
	interface->print();
	delete(interface);
}

float Vector::compareTo(Vector* other)
{
	Interface* interface = toInterface();
	Interface* otherInterface = other->toInterface();

	float toReturn = interface->compareTo(otherInterface);

	delete(interface);
	delete(otherInterface);
	return toReturn;
}

void Vector::random(float range)
{
	Interface* interface = this->toInterface();
	interface->random(range);
	this->copyFrom(interface);
	delete(interface);
}

void Vector::transposeMatrix(unsigned width)
{
	Interface* interface = this->toInterface();
	interface->transposeMatrix(width);
	this->copyFrom(interface);
	delete(interface);
}

