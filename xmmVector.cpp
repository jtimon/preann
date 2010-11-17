#include "xmmVector.h"

XmmVector::XmmVector(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

	size_t byteSize = getByteSize(size, vectorType);
	data = mi_malloc(byteSize);

	switch (vectorType){

	case BYTE:
		SetValueToAnArray<unsigned char>(data, byteSize, 128);
		break;
	case FLOAT:
		SetValueToAnArray<float>(data, byteSize/sizeof(float), 0);
		break;
	case BIT:
	case SIGN:
		SetValueToAnArray<unsigned char>(data, byteSize, 0);
		break;
	}
}

XmmVector::~XmmVector()
{
	if (data) {
		mi_free(data);
		data = NULL;
	}
}

Vector* XmmVector::clone()
{
	Vector* clone = new XmmVector(size, vectorType);
	copyToVector(clone);
	return clone;
}

void XmmVector::bitCopyFrom(Interface *interface, unsigned char *vectorData)
{
    unsigned blockOffset = 0;
    unsigned bytePos = 0;
    unsigned char vectorMask = 128;
    for (unsigned i=0; i < size; i++){

		if (interface->getElement(i) > 0){
			vectorData[blockOffset + bytePos] |= vectorMask;
		} else {
			vectorData[blockOffset + bytePos] &= ~vectorMask;
		}

		if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK-1)){
			bytePos = 0;
			if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK-1)){
				blockOffset += BYTES_PER_BLOCK;
				vectorMask = 128;
			} else {
				vectorMask >>= 1;
			}
		} else {
			++bytePos;
		}
	}
}

void XmmVector::copyFromImpl(Interface* interface)
{
	switch (vectorType){
	case BYTE:
		for (unsigned i=0; i < size; i++){
			((unsigned char*)(data))[i] = interface->getElement(i);
		}
		break;
	case FLOAT:
		for(unsigned i = 0;i < size;i++){
			((float*)(data))[i] = interface->getElement(i);
		}
		break;
	case BIT:
	case SIGN:
		unsigned char *vectorData = (unsigned char*)(data);
		bitCopyFrom(interface, vectorData);
		break;
	}
}

void XmmVector::bitCopyTo(unsigned char *vectorData, Interface *interface)
{
    unsigned blockOffset = 0;
    unsigned bytePos = 0;
    unsigned char vectorMask = 128;
    for (unsigned i=0; i < size; i++){

		if (vectorData[blockOffset + bytePos] & vectorMask){
			interface->setElement(i, 1);
		} else {
			interface->setElement(i, 0);
		}

		if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK-1)){
			bytePos = 0;
			if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK-1)){
				blockOffset += BYTES_PER_BLOCK;
				vectorMask = 128;
			} else {
				vectorMask >>= 1;
			}
		} else {
			++bytePos;
		}
	}
}

void XmmVector::copyToImpl(Interface* interface)
{
	switch (vectorType){
	case BYTE:
		for (unsigned i=0; i < size; i++){
			interface->setElement(i, ((unsigned char*)data)[i]);
		}
		break;
	case FLOAT:
		for (unsigned i=0; i < size; i++){
			interface->setElement(i, ((float*)data)[i]);
		}
		break;
	case BIT:
	case SIGN:
		unsigned char* vectorData = (unsigned char*)(data);
		bitCopyTo(vectorData, interface);
		break;
	}
}

void XmmVector::activation(Vector* resultsVect, FunctionType functionType)
{
	float* results = (float*)resultsVect->getDataPointer();

	switch (vectorType){
	case BYTE:
		{
			std::string error = "XmmVector::activation is not implemented for VectorType BYTE.";
			throw error;
		}break;
	case FLOAT:
		{
			for (unsigned i=0; i < size; i++){
				((float*)data)[i] = Function(results[i], functionType);
			}
		}
		break;
	case BIT:
	case SIGN:
		{
			unsigned char* vectorData = (unsigned char*)data;

			unsigned blockOffset = 0;
			unsigned bytePos = 0;
			unsigned char vectorMask = 128;

			for (unsigned i=0; i < size; i++){

				if (results[i] > 0){
					vectorData[blockOffset + bytePos] |= vectorMask;
				} else {
					vectorData[blockOffset + bytePos] &= ~vectorMask;
				}

				if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK-1)){
					bytePos = 0;
					if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK-1)){
						blockOffset += BYTES_PER_BLOCK;
						vectorMask = 128;
					} else {
						vectorMask >>= 1;
					}
				} else {
					++bytePos;
				}
			}
		}
	}
}

//TODO D esto es igual en CppVector
void XmmVector::mutate(unsigned pos, float mutation)
{
	if (pos > size){
		std::string error = "The position being mutated is greater than the size of the vector.";
		throw error;
	}
	switch (vectorType){
	case BYTE:{
		unsigned char* weigh = &(((unsigned char*)data)[pos]);
		int result = (int)mutation + *weigh;
		if (result <= 0){
			*weigh = 0;
		}
		else if (result >= 255) {
			*weigh = 255;
		}
		else {
			*weigh = result;
		}
		}break;
	case FLOAT:
		((float*)data)[pos] += mutation;
		break;
	case BIT:
	case SIGN:
		{
		std::string error = "XmmVector::mutate is not implemented for VectorType BIT nor SIGN.";
		throw error;
		}
	}
}

//TODO D esto es igual en CppVector
void XmmVector::crossoverImpl(Vector* other, Interface* bitVector)
{
	void* otherWeighs = other->getDataPointer();
	void* thisWeighs = this->getDataPointer();

	switch (vectorType){
	case BYTE:{
		unsigned char auxWeigh;

		for (unsigned i=0; i < size; i++){

			if (bitVector->getElement(i)){
				auxWeigh = ((unsigned char*)thisWeighs)[i];
				((unsigned char*)thisWeighs)[i] = ((unsigned char*)otherWeighs)[i];
				((unsigned char*)otherWeighs)[i] = auxWeigh;
			}
		}
		}break;
	case FLOAT:
		float auxWeigh;

		for (unsigned i=0; i < size; i++){

			if (bitVector->getElement(i)){
				auxWeigh = ((float*)thisWeighs)[i];
				((float*)thisWeighs)[i] = ((float*)otherWeighs)[i];
				((float*)otherWeighs)[i] = auxWeigh;
			}
		}
		break;
	case BIT:
	case SIGN:
		{
		std::string error = "XmmConnection::weighCrossover is not implemented for VectorType BIT nor SIGN.";
		throw error;
		}
	}
}

unsigned XmmVector::getByteSize(unsigned size, VectorType vectorType)
{
	unsigned numBlocks;
	switch (vectorType){
	case BYTE:
		numBlocks = ((size-1)/BYTES_PER_BLOCK)+1;
		break;
	case FLOAT:
		numBlocks = ((size-1)/FLOATS_PER_BLOCK)+1;
		break;
	case BIT:
	case SIGN:
		numBlocks = ((size-1)/BITS_PER_BLOCK)+1;
		break;
	}
	return numBlocks * BYTES_PER_BLOCK;
}
