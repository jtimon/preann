
#include "interface.h"

Interface::Interface(unsigned  size, VectorType vectorType=FLOAT)
{
	this->size = size;
	this->vectorType = vectorType;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	if (vectorType == FLOAT){

		for (unsigned i=0; i< size/sizeof(float); i++){
			((float*)data)[i] = 0;
		}
	} else {
		for (unsigned i=0; i < byteSize; i++){
			((unsigned char*)data)[i] = 1;
		}
	}
}

Interface::~Interface()
{
	mi_free(data);
}

void* Interface::getDataPointer()
{
	return data;
}

unsigned Interface::getByteSize()
{
	if (vectorType == FLOAT){
		return size * sizeof(float);
	}
	return ((size - 1)/BITS_PER_BYTE) + 1;
}

VectorType Interface::getVectorType()
{
	return vectorType;
}

unsigned Interface::getSize()
{
	return size;
}

float Interface::getElement(unsigned  pos)
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

		unsigned char mask = (unsigned char)( 128>>(pos % BITS_PER_BYTE) );

		if ( ((unsigned char*)data)[pos / BITS_PER_BYTE] & mask){
			return 1;
		}
		if (vectorType == BIT) {
			return 0;
		}
		return -1;
	}
}

void Interface::setElement(unsigned  pos, float value)
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
		unsigned char mask = (unsigned char)( 128>>(pos % BITS_PER_BYTE) );

		if (value > 0){
			((unsigned char*)data)[pos / BITS_PER_BYTE] |= mask;
		} else {
			((unsigned char*)data)[pos / BITS_PER_BYTE] &= ~mask;
		}
	}
}

float Interface::compareTo(Interface *other)
{
	float accumulator = 0;
	for (unsigned i=0; i < this->size; i++) {
		accumulator += this->getElement(i) - other->getElement(i);
	}
	return accumulator;
}

void Interface::print()
{
	cout<<"----------------"<<endl;
	for (unsigned i=0; i < size; i++){
		cout<<getElement(i)<<" ";
	}
	cout<<endl<<"----------------"<<endl;
}


