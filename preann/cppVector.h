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
	unsigned getByteSize()
	{
		switch (vectorTypeTempl){
			case BIT:
			case SIGN:
				return (((tSize-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
			default:
				return tSize * sizeof(c_typeTempl);
		}
	}

	virtual void copyFromImpl(Interface* interface)
	{
		memcpy(data, interface->getDataPointer(), interface->getByteSize());
	}

	virtual void copyToImpl(Interface* interface)
	{
		memcpy(interface->getDataPointer(), data, this->getByteSize());
	}
public:

	virtual ImplementationType getImplementationType() {
		return C;
	};
	CppVector(){};

	CppVector(unsigned size)
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

	~CppVector()
	{
		if (data) {
			mi_free(data);
			data = NULL;
		}
	}

	virtual Vector* clone()
	{
		Vector* clone = new CppVector<vectorTypeTempl, c_typeTempl>(tSize);
		copyTo(clone);
		return clone;
	}

	virtual void activation(Vector* resultsVect, FunctionType functionType)
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
					((c_typeTempl*)data)[i] = Function<c_typeTempl>(results[i], functionType);
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

	virtual void mutateImpl(unsigned pos, float mutation)
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

	virtual void crossoverImpl(Vector* other, Interface* bitVector)
	{
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

};

#endif /* CPPVECTOR_H_ */
