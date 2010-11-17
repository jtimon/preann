
#include "cppVector.h"

CppVector::CppVector(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	switch (vectorType){
		case BYTE:  SetValueToAnArray<unsigned char>(data, byteSize, 128); 		break;
		case FLOAT: SetValueToAnArray<float>(data, byteSize/sizeof(float), 0);  break;
		case BIT:
		case SIGN: 	SetValueToAnArray<unsigned char>(data, byteSize, 0);		break;
	}
}

CppVector::~CppVector()
{
	if (data) {
		mi_free(data);
		data = NULL;
	}
}

Vector* CppVector::clone()
{
	Vector* clone = new CppVector(size, vectorType);
	copyTo(clone);
	return clone;
}

void CppVector::copyFromImpl(Interface* interface)
{
	memcpy(data, interface->getDataPointer(), interface->getByteSize());
}

void CppVector::copyToImpl(Interface* interface)
{
	memcpy(interface->getDataPointer(), data, this->getByteSize());
}

void CppVector::activation(Vector* resultsVect, FunctionType functionType)
{
	float* results = (float*)resultsVect->getDataPointer();

	switch (vectorType){
	case BYTE:
		{
			std::string error = "CppVector::activation is not implemented for VectorType BYTE.";
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
			unsigned* vectorData = (unsigned*)data;
			unsigned mask;
			for (unsigned i=0; i < size; i++){

				if (i % BITS_PER_UNSIGNED == 0){
					mask = 0x80000000;
				} else {
					mask >>= 1;
				}

				if (results[i] > 0){
					vectorData[i/BITS_PER_UNSIGNED] |= mask;
				} else {
					vectorData[i/BITS_PER_UNSIGNED] &= ~mask;
				}
			}
		}
	}
}

void CppVector::mutateImpl(unsigned pos, float mutation)
{
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
		unsigned mask = 0x80000000>>(pos % BITS_PER_UNSIGNED) ;
		((unsigned*)data)[pos / BITS_PER_UNSIGNED] ^= mask;
		}
	}
}
void CppVector::crossoverImpl(Vector* other, Interface* bitVector)
{
	if (size != other->getSize()){
		std::string error = "The Connections must have the same size to crossover them.";
		throw error;
	}
	if (vectorType != other->getVectorType()){
		std::string error = "The Connections must have the same type to crossover them.";
		throw error;
	}

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
		std::string error = "CppVector::weighCrossover is not implemented for VectorType BIT nor SIGN.";
		throw error;
		}
	}
}


unsigned CppVector::getByteSize()
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
