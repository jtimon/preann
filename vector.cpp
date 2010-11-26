/*
 * vector.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "vector.h"

void Vector::copyFromInterface(Interface* interface)
{
	if (getSize() < interface->getSize()){
		std::string error = "The Interface is greater than the Vector.";
		throw error;
	}
	if (getVectorType() != interface->getVectorType()){
		std::string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}
	copyFromImpl(interface);
}

void Vector::copyToInterface(Interface* interface)
{
	if (interface->getSize() < getSize()){
		std::string error = "The Vector is greater than the Interface.";
		throw error;
	}
	if (getVectorType() != interface->getVectorType()){
		std::string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}
	copyToImpl(interface);
}

void* Vector::getDataPointer()
{
	return data;
}

unsigned Vector::getSize()
{
	return tSize;
}

Interface* Vector::toInterface()
{
	Interface* toReturn = new Interface(getSize(), this->getVectorType());
	this->copyToImpl(toReturn);
	return toReturn;
}

void Vector::copyFrom(Vector* vector)
{
	Interface* interface = vector->toInterface();
	this->copyFromInterface(interface);
	delete(interface);
}

void Vector::copyTo(Vector* vector)
{
	Interface* interface = this->toInterface();
	vector->copyFromInterface(interface);
	delete(interface);
}

void Vector::save(FILE* stream)
{
	Interface* interface = toInterface();
	interface->save(stream);
	delete(interface);
}

void Vector::load(FILE* stream)
{
	Interface interface(tSize, getVectorType());
	interface.load(stream);
	copyFromInterface(&interface);
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
	this->copyFromImpl(interface);
	delete(interface);
}

