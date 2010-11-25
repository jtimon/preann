/*
 * xmmVector.h
 *
 *  Created on: Nov 17, 2009
 *      Author: timon
 */

#ifndef XMMVECTOR_H_
#define XMMVECTOR_H_

#include "vectorImpl.h"
#include "sse2_code.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class XmmVector: virtual public Vector, virtual public VectorImpl<vectorTypeTempl, c_typeTempl> {
protected:
	static unsigned getByteSize(unsigned size, VectorType vectorType);
    void bitCopyFrom(Interface *interface, unsigned char *vectorData);
    void bitCopyTo(unsigned char *vectorData, Interface *interface);
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
public:
    XmmVector() {};
	XmmVector(unsigned size);
	virtual ~XmmVector();
	virtual ImplementationType getImplementationType() {
		return SSE2;
	};

	virtual Vector* clone();

	virtual void activation(Vector* results, FunctionType functionType);
	//for weighs
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);

};

template <VectorType vectorTypeTempl, class c_typeTempl>
XmmVector<vectorTypeTempl, c_typeTempl>::XmmVector(unsigned size)
{
	this->tSize = size;

	size_t byteSize = getByteSize(size, vectorTypeTempl);
	data = mi_malloc(byteSize);

	switch (vectorTypeTempl){

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

template <VectorType vectorTypeTempl, class c_typeTempl>
XmmVector<vectorTypeTempl, c_typeTempl>::~XmmVector()
{
	if (data) {
		mi_free(data);
		data = NULL;
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
Vector* XmmVector<vectorTypeTempl, c_typeTempl>::clone()
{
	Vector* clone = new XmmVector<vectorTypeTempl, c_typeTempl>(tSize);
	copyTo(clone);
	return clone;
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmVector<vectorTypeTempl, c_typeTempl>::bitCopyFrom(Interface *interface, unsigned char *vectorData)
{
    unsigned blockOffset = 0;
    unsigned bytePos = 0;
    unsigned char vectorMask = 128;
    for (unsigned i=0; i < tSize; i++){

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

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmVector<vectorTypeTempl, c_typeTempl>::copyFromImpl(Interface* interface)
{
	switch (vectorTypeTempl) {
		default:
			memcpy(data, interface->getDataPointer(), interface->getByteSize());
			break;
		case BIT:
		case SIGN:
		    unsigned blockOffset = 0;
		    unsigned bytePos = 0;
		    unsigned char vectorMask = 128;
		    unsigned char *vectorData = (unsigned char*)(data);

		    for (unsigned i=0; i < tSize; i++){
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
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmVector<vectorTypeTempl, c_typeTempl>::bitCopyTo(unsigned char *vectorData, Interface *interface)
{
    unsigned blockOffset = 0;
    unsigned bytePos = 0;
    unsigned char vectorMask = 128;
    for (unsigned i=0; i < tSize; i++){

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

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmVector<vectorTypeTempl, c_typeTempl>::copyToImpl(Interface* interface)
{
	switch (vectorTypeTempl){
	case BYTE:
		for (unsigned i=0; i < tSize; i++){
			interface->setElement(i, ((unsigned char*)data)[i]);
		}
		break;
	case FLOAT:
		for (unsigned i=0; i < tSize; i++){
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

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmVector<vectorTypeTempl, c_typeTempl>::activation(Vector* resultsVect, FunctionType functionType)
{
	float* results = (float*)resultsVect->getDataPointer();

	switch (vectorTypeTempl){
	case BYTE:
		{
			std::string error = "XmmVector::activation is not implemented for VectorType BYTE.";
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
			unsigned char* vectorData = (unsigned char*)data;

			unsigned blockOffset = 0;
			unsigned bytePos = 0;
			unsigned char vectorMask = 128;

			for (unsigned i=0; i < tSize; i++){

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
template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmVector<vectorTypeTempl, c_typeTempl>::mutateImpl(unsigned pos, float mutation)
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
		std::string error = "XmmVector::mutate is not implemented for VectorType BIT nor SIGN.";
		throw error;
		}
	}
}

//TODO D esto es igual en CppVector
template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmVector<vectorTypeTempl, c_typeTempl>::crossoverImpl(Vector* other, Interface* bitVector)
{
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
		std::string error = "XmmVector::crossoverImpl is not implemented for VectorType BIT nor SIGN.";
		throw error;
		}
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
unsigned XmmVector<vectorTypeTempl, c_typeTempl>::getByteSize(unsigned size, VectorType vectorType)
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


#endif /* XMMVECTOR_H_ */
