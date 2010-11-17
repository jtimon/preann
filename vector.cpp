/*
 * vector.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "vector.h"

void Vector::mutate(unsigned pos, float mutation)
{
	if (pos > size){
		std::string error = "The position being mutated is greater than the size of the vector.";
		throw error;
	}
	mutateImpl(pos, mutation);
}

void Vector::crossover(Vector* other, Interface* bitVector)
{
	if (size != other->getSize()){
		std::string error = "The Vectors must have the same size to crossover them.";
		throw error;
	}
	if (vectorType != other->getVectorType()){
		std::string error = "The Vectors must have the same type to crossover them.";
		throw error;
	}
	crossoverImpl(other, bitVector);
}

void Vector::copyFromInterface(Interface* interface)
{
	if (size < interface->getSize()){
		std::string error = "The Interface is greater than the Vector.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
		std::string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}
	copyFromImpl(interface);
}

void Vector::copyToInterface(Interface* interface)
{
	if (interface->getSize() < size){
		std::string error = "The Vector is greater than the Interface.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
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
	return size;
}

VectorType Vector::getVectorType()
{
	return vectorType;
}

Interface* Vector::toInterface()
{
	Interface* toReturn = new Interface(this->size, this->vectorType);
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

