/*
 * cppVector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef CPPVECTOR_H_
#define CPPVECTOR_H_

#include "vectorImpl.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class CppVector: virtual public Vector, virtual public VectorImpl<vectorTypeTempl, c_typeTempl> {
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

template <VectorType vectorTypeTempl, class c_typeTempl>
CppVector<vectorTypeTempl, c_typeTempl>::CppVector(unsigned size)
{
	this->tSize = size;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	switch (vectorTypeTempl){
		case BYTE:
			SetValueToAnArray<c_typeTempl>(data, byteSize/sizeof(c_typeTempl), 128);
			break;
		default:
			SetValueToAnArray<c_typeTempl>(data, byteSize/sizeof(c_typeTempl), 0);
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
CppVector<vectorTypeTempl, c_typeTempl>::~CppVector()
{
	if (data) {
		mi_free(data);
		data = NULL;
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
Vector* CppVector<vectorTypeTempl, c_typeTempl>::clone()
{
	Vector* clone = new CppVector<vectorTypeTempl, c_typeTempl>(tSize);
	copyTo(clone);
	return clone;
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CppVector<vectorTypeTempl, c_typeTempl>::copyFromImpl(Interface* interface)
{
	memcpy(data, interface->getDataPointer(), interface->getByteSize());
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CppVector<vectorTypeTempl, c_typeTempl>::copyToImpl(Interface* interface)
{
	memcpy(interface->getDataPointer(), data, this->getByteSize());
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CppVector<vectorTypeTempl, c_typeTempl>::activation(Vector* resultsVect, FunctionType functionType)
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

template <VectorType vectorTypeTempl, class c_typeTempl>
void CppVector<vectorTypeTempl, c_typeTempl>::mutateImpl(unsigned pos, float mutation)
{
	switch (vectorTypeTempl){
	case BYTE:{
		c_typeTempl* weigh = &(((c_typeTempl*)data)[pos]);
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
		((c_typeTempl*)data)[pos] += mutation;
		break;
	case BIT:
	case SIGN:
		{
		unsigned mask = 0x80000000>>(pos % BITS_PER_UNSIGNED) ;
		((unsigned*)data)[pos / BITS_PER_UNSIGNED] ^= mask;
		}
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CppVector<vectorTypeTempl, c_typeTempl>::crossoverImpl(Vector* other, Interface* bitVector)
{
	if (tSize != other->getSize()){
		std::string error = "The Connections must have the same size to crossover them.";
		throw error;
	}
	if (vectorTypeTempl != other->getVectorType()){
		std::string error = "The Connections must have the same type to crossover them.";
		throw error;
	}
	switch (vectorTypeTempl){
		case BIT:
		case SIGN:
			{
			std::string error = "CppVector::crossoverImpl is not implemented for VectorType BIT nor SIGN.";
			throw error;
			}
		default:
		{
			c_typeTempl* otherWeighs = (c_typeTempl*)other->getDataPointer();
			c_typeTempl* thisWeighs = (c_typeTempl*)this->getDataPointer();
			c_typeTempl auxWeigh;

			for (unsigned i=0; i < tSize; i++){
				if (bitVector->getElement(i)){
					auxWeigh = thisWeighs[i];
					thisWeighs[i] = otherWeighs[i];
					otherWeighs[i] = auxWeigh;
				}
			}
		}
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
unsigned CppVector<vectorTypeTempl, c_typeTempl>::getByteSize()
{
	switch (vectorTypeTempl){
		case BIT:
		case SIGN:
			return (((tSize-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
		default:
			return tSize * sizeof(c_typeTempl);
	}
}

#endif /* CPPVECTOR_H_ */
