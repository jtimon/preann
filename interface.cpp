
#include "interface.h"

Interface::Interface()
{
	this->size = 0;
	data = NULL;
}

Interface::Interface(FILE* stream)
{
	fread(&size, sizeof(unsigned), 1, stream);
	fread(&vectorType, sizeof(VectorType), 1, stream);
	data = mi_malloc(getByteSize());
	fread(data, getByteSize(), 1, stream);
}

Interface::Interface(unsigned  size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	switch (vectorType){
	case BYTE:
		for (unsigned i=0; i < byteSize; i++){
			((unsigned char*)data)[i] = 128;
		}
		break;
	case FLOAT:
		for (unsigned i=0; i< size; i++){
			((float*)data)[i] = 0;
		}
		break;
	case BIT:
	case SIGN:
		for (unsigned i=0; i < byteSize; i++){
			((unsigned char*)data)[i] = 0;
		}
	}
}

Interface::Interface(Interface* toCopy)
{
	this->size = toCopy->getSize();
	this->vectorType = toCopy->getVectorType();

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);
	memcpy(data, toCopy->getDataPointer(), byteSize);
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
	switch (vectorType){
	case BYTE:
		return size;
	case FLOAT:
		return size * sizeof(float);
	case BIT:
	case SIGN:
		return (((size-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	}
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
		std::string error = buffer;
		throw error;
	}
	switch (vectorType){
	case BYTE:
		return ((unsigned char*)data)[pos];
	case FLOAT:
		return ((float*)data)[pos];
	case BIT:
	case SIGN:
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
		std::string error = buffer;
		throw error;
	}
	switch (vectorType){
	case BYTE:
		((unsigned char*)data)[pos] = (unsigned char)value;
		break;
	case FLOAT:
		((float*)data)[pos] = value;
		break;
	case BIT:
	case SIGN:
		unsigned mask = 0x80000000>>(pos % BITS_PER_UNSIGNED) ;

		if (value > 0){
			((unsigned*)data)[pos / BITS_PER_UNSIGNED] |= mask;
		} else {
			((unsigned*)data)[pos / BITS_PER_UNSIGNED] &= ~mask;
		}
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

void Interface::random(float range)
{
	switch (vectorType){
	case BYTE:
		unsigned charRange;
		if (range >= 128){
			charRange = 127;
		} else {
			charRange = (unsigned)range;
		}
		for (unsigned i=0; i < size; i++){
			setElement(i, 128 + (unsigned char)randomInt(charRange));
		}
		break;
	case FLOAT:
		for (unsigned i=0; i < size; i++){
			setElement(i, randomFloat(range));
		}
		break;
	case BIT:
	case SIGN:
		for (unsigned i=0; i < size; i++){
			setElement(i, randomUnsigned(2));
		}
		break;
	}
}

void Interface::save(FILE* stream)
{
	fwrite(&size, sizeof(unsigned), 1, stream);
	fwrite(&vectorType, sizeof(VectorType), 1, stream);
	fwrite(data, getByteSize(), 1, stream);
}

void Interface::load(FILE* stream)
{
	unsigned size2;
	VectorType vectorType2;
	fread(&size2, sizeof(unsigned), 1, stream);
	fread(&vectorType2, sizeof(VectorType), 1, stream);

	if (size2 != size){
		std::string error = "The size of the Interface is different than the size to load.";
		throw error;
	}
	if (vectorType2 != vectorType){
		std::string error = "The Type of the Interface is different than the Vector Type to load.";
		throw error;
	}
	data = mi_malloc(getByteSize());
	fread(data, getByteSize(), 1, stream);
}

void Interface::print()
{
	printf("----------------\n", 1);
	for (unsigned i=0; i < size; i++){
		switch (vectorType){
		case BYTE:
			printf("%d ", (int)((unsigned char)getElement(i) - 128));
			break;
		case FLOAT:
			printf("%f ", getElement(i));
			break;
		case BIT:
		case SIGN:
			printf("%d ", (int)getElement(i));
		}
	}
	printf("\n----------------\n", 1);
}

void Interface::transposeMatrix(unsigned width)
{
	if (size % width != 0) {
		char buffer[100];
		sprintf(buffer, "The interface cannot be a matrix of witdth %d, it have size %d.", width, size);
		std::string error = buffer;
		throw error;
	}

	Interface aux(this);

	unsigned height = size / width;
	for (unsigned i=0; i < width; i++){
		for (unsigned j=0; j < height; j++){
			setElement((i * height) + j, aux.getElement(i + (j * width)));
		}
	}
}
