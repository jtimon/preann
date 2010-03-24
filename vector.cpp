/*
 * vector.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "vector.h"

Vector::Vector(unsigned size, VectorType vectorType = FLOAT)
{
	this->size = size;
	this->vectorType = vectorType;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	if (vectorType == FLOAT){

		unsigned floatSize = byteSize/sizeof(float);
		for (unsigned i=0; i< floatSize; i++){
			((float*)data)[i] = 0;
		}
	}
	else {

		for (unsigned i=0; i < byteSize/sizeof(unsigned); i++){
			((unsigned*)data)[i] = 1;
		}
	}
}

Vector::~Vector()
{
	if (data) {
		mi_free(data);
	}
}

void* Vector::getDataPointer()
{
	return data;
}

unsigned Vector::getByteSize()
{
	if (vectorType == FLOAT){

		return (((size-1)/FLOATS_PER_BLOCK)+1) * FLOATS_PER_BLOCK * sizeof(float);
	}
	else {
		return (((size-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	}
}

unsigned Vector::getWeighsSize()
{
	if (vectorType == FLOAT){
		return (((size-1)/FLOATS_PER_BLOCK)+1) * FLOATS_PER_BLOCK;
	}
	else {
		return (((size-1)/BITS_PER_UNSIGNED)+1) * BITS_PER_UNSIGNED;
	}
}

unsigned Vector::getSize()
{
	return size;
}

VectorType Vector::getVectorType()
{
	return vectorType;
}

unsigned Vector::posToBitPos(unsigned  pos)
{
	return pos%BITS_PER_UNSIGNED;
}

unsigned Vector::posToUnsignedPos(unsigned  pos)
{
	return pos/BITS_PER_UNSIGNED;
}

float Vector::compareTo(Vector *other)
{
	float accumulator = 0;
	for (unsigned i=0; i < this->size; i++) {
		accumulator += this->getElement(i) - other->getElement(i);
	}
	return accumulator;
}

void Vector::setElement(unsigned  pos, float value)
{
	if (pos >= size){
		char buffer[100];
		sprintf(buffer, "Cannot set the element in position %d: the size of the vector is %d.", pos, size);
		string error = buffer;
		throw error;
	}

	if (vectorType == FLOAT){

		((float*)data)[pos] = value;

	} else {
		unsigned unsignedPos = posToUnsignedPos(pos);
		unsigned bitPos = posToBitPos(pos);
		unsigned mask = (unsigned)(0x80000000>>bitPos);
		if (value > 0){
			((unsigned*)data)[unsignedPos] |= mask;
		} else if (value == 0 || value == -1) {
			((unsigned*)data)[unsignedPos] &= ~mask;
		}
	}
}

float Vector::getElement(unsigned  pos)
{
	if (pos >= size){
		char buffer[100];
		sprintf(buffer, "Cannot get the element in position %d: the size of the vector is %d.", pos, size);
		string error = buffer;
		throw error;
	}


	if (vectorType == FLOAT){
		return ((float*)data)[pos];
	}
	else {
		unsigned unsignedPos = posToUnsignedPos(pos);
		unsigned bitPos = posToBitPos(pos);

		unsigned mask = (unsigned)(0x80000000>>bitPos);
		if (((unsigned*)data)[unsignedPos] & mask){
			return 1;
		}
		if (vectorType == BIT) {
			return 0;
		}
		return -1;
	}
}

void Vector::showVector()
{
	for (unsigned i=0; i < size; i++){
		cout<<getElement(i)<<" ";
		//printf("%f ", getElement(i))
		//if (i % BITS_PER_UNSIGNED == BITS_PER_UNSIGNED - 1) cout<<endl;
	}
	cout<<endl<<"----------------"<<endl;
}

