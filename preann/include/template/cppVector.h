/*
 * cppVector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef CPPVECTOR_H_
#define CPPVECTOR_H_
#ifdef CPP_IMPL

#include "vector.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class CppVector: virtual public Vector{
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

	virtual VectorType getVectorType()
	{
		return vectorTypeTempl;
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

};

#endif
#endif /* CPPVECTOR_H_ */
