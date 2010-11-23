/*
 * cppVector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef CPPVECTOR_H_
#define CPPVECTOR_H_

#include "vectorImpl.h"

template <VectorType vectorTypeTempl>
class CppVector: virtual public Vector, virtual public VectorImpl<vectorTypeTempl> {
protected:
	unsigned getByteSize();
	virtual void copyToImpl(Interface* interface);
	virtual void copyFromImpl(Interface* interface);
public:
	CppVector(){};
	CppVector(unsigned size);
	virtual ~CppVector();
	virtual ImplementationType getImplementationType() {
		return C;
	};

	virtual Vector* clone();
	virtual void activation(Vector* results, FunctionType functionType);
	//for weighs
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);

};

template <VectorType vectorTypeTempl>
CppVector<vectorTypeTempl>::CppVector(unsigned size)
{
	this->tSize = size;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	switch (vectorTypeTempl){
		case BYTE:  SetValueToAnArray<unsigned char>(data, byteSize, 128); 		break;
		case FLOAT: SetValueToAnArray<float>(data, byteSize/sizeof(float), 0);  break;
		case BIT:
		case SIGN: 	SetValueToAnArray<unsigned char>(data, byteSize, 0);		break;
	}
}

template <VectorType vectorTypeTempl>
CppVector<vectorTypeTempl>::~CppVector()
{
	if (data) {
		mi_free(data);
		data = NULL;
	}
}

template <VectorType vectorTypeTempl>
Vector* CppVector<vectorTypeTempl>::clone()
{
	Vector* clone = new CppVector<vectorTypeTempl>(tSize);
	copyTo(clone);
	return clone;
}

template <VectorType vectorTypeTempl>
void CppVector<vectorTypeTempl>::copyFromImpl(Interface* interface)
{
	memcpy(data, interface->getDataPointer(), interface->getByteSize());
}

template <VectorType vectorTypeTempl>
void CppVector<vectorTypeTempl>::copyToImpl(Interface* interface)
{
	memcpy(interface->getDataPointer(), data, this->getByteSize());
}

template <VectorType vectorTypeTempl>
void CppVector<vectorTypeTempl>::activation(Vector* resultsVect, FunctionType functionType)
{
	float* results = (float*)resultsVect->getDataPointer();

	switch (vectorTypeTempl){
	case BYTE:
		{
			std::string error = "CppVector::activation is not implemented for VectorType BYTE.";
			throw error;
		}break;
	case FLOAT:
		{
			for (unsigned i=0; i < tSize; i++){
				((float*)data)[i] = Function(results[i], functionType);
			}
		}
		break;
	case BIT:
	case SIGN:
		{
			unsigned* vectorData = (unsigned*)data;
			unsigned mask;
			for (unsigned i=0; i < tSize; i++){

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

template <VectorType vectorTypeTempl>
void CppVector<vectorTypeTempl>::mutateImpl(unsigned pos, float mutation)
{
	switch (vectorTypeTempl){
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

template <VectorType vectorTypeTempl>
void CppVector<vectorTypeTempl>::crossoverImpl(Vector* other, Interface* bitVector)
{
	if (tSize != other->getSize()){
		std::string error = "The Connections must have the same size to crossover them.";
		throw error;
	}
	if (vectorTypeTempl != other->getVectorType()){
		std::string error = "The Connections must have the same type to crossover them.";
		throw error;
	}

	void* otherWeighs = other->getDataPointer();
	void* thisWeighs = this->getDataPointer();

	switch (vectorTypeTempl){
	case BYTE:{
		unsigned char auxWeigh;

		for (unsigned i=0; i < tSize; i++){

			if (bitVector->getElement(i)){
				auxWeigh = ((unsigned char*)thisWeighs)[i];
				((unsigned char*)thisWeighs)[i] = ((unsigned char*)otherWeighs)[i];
				((unsigned char*)otherWeighs)[i] = auxWeigh;
			}
		}
		}break;
	case FLOAT:
		float auxWeigh;

		for (unsigned i=0; i < tSize; i++){

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

template <VectorType vectorTypeTempl>
unsigned CppVector<vectorTypeTempl>::getByteSize()
{
	switch (vectorTypeTempl){
	case BYTE:
		return tSize;
	case FLOAT:
		return tSize * sizeof(float);
	case BIT:
	case SIGN:
		return (((tSize-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	}
}

#endif /* CPPVECTOR_H_ */
