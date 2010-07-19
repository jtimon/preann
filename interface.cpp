
#include "interface.h"

Interface::Interface(unsigned  size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	if (vectorType == FLOAT){

		for (unsigned i=0; i< size; i++){
			((float*)data)[i] = 0;
		}
	} else {
		for (unsigned i=0; i < byteSize/sizeof(unsigned); i++){
			((unsigned*)data)[i] = 0;
		}
	}
}

Interface::Interface(Interface* toCopy)
{
	this->size = toCopy->getSize();
	this->vectorType = toCopy->getVectorType();

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	for (unsigned i = 0; i < size; i++){
		this->setElement(i, toCopy->getElement(i));
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
	return (((size - 1)/BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
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

		//TODO quitar mensaje
		printf("obteniendo un elemento que no es float");
		unsigned  mask = 0x80000000>>(pos % BITS_PER_UNSIGNED) ;

		if ( ((unsigned*)data)[pos / BITS_PER_UNSIGNED] & mask){
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
		//TODO quitar mensaje
		printf("informando un elemento que no es float");
		unsigned mask = 0x80000000>>(pos % BITS_PER_UNSIGNED) ;

		if (value > 0){
			((unsigned*)data)[pos / BITS_PER_UNSIGNED] |= mask;
		} else {
			((unsigned*)data)[pos / BITS_PER_UNSIGNED] &= ~mask;
		}
		//printf("pos %d vectorPos %d bitPos %d valor %d data %d mask %d almacenado %d \n", pos, pos / BITS_PER_BYTE, pos % BITS_PER_BYTE, (int)value, ((unsigned*)data)[pos / BITS_PER_BYTE], mask, ((unsigned*)data)[pos / BITS_PER_BYTE] & mask);
	}
}

float Interface::compareTo(Interface *other)
{
	float accumulator = 0;
	for (unsigned i=0; i < this->size; i++) {
		float difference = this->getElement(i) - other->getElement(i);
		if (difference > 0){
			accumulator += difference;
		} else {
			accumulator -= difference;
		}

	}
	return accumulator;
}

void Interface::setRandomBits(unsigned num)
{
	for (unsigned i=0; i < num; i++){
		setElement(randomUnsigned(size), 1);
	}
}

void Interface::print()
{
	printf("----------------\n", 1);
	for (unsigned i=0; i < size; i++){
		if (vectorType == FLOAT){
			printf("%f ", getElement(i));
		} else {
			printf("%d ", (int)getElement(i));
		}
	}
	printf("\n----------------\n", 1);
}


