/*
 * cppVector.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "cppVector.h"

CppVector::CppVector(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

	size_t byteSize = getByteSize();
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

CppVector::~CppVector()
{
	if (data) {
		mi_free(data);
		data = NULL;
	}
}

void CppVector::copyFrom(Interface* interface)
{
	if (size < interface->getSize()){
		string error = "The Interface is greater than the Vector.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
		string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}
	memcpy(data, interface->getDataPointer(), interface->getByteSize());
}

void CppVector::copyTo(Interface* interface)
{
	if (interface->getSize() < size){
		string error = "The Vector is greater than the Interface.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
		string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}
	memcpy(interface->getDataPointer(), data, this->getByteSize());
}

void CppVector::activation(float* results, FunctionType functionType)
{
	if (vectorType == FLOAT){
		for (unsigned i=0; i < size; i++){
			//TODO quitar mensaje
			printf(" %f ", results[i]);
			((float*)data)[i] = Function(results[i], functionType);
		}
		//TODO quitar mensaje
		printf("\n");
	} else {
		unsigned* vectorData = (unsigned*)data;
		unsigned mask;
		for (unsigned i=0; i < size; i++){

			if (i % BITS_PER_UNSIGNED == 0){
				mask = 0x80000000;
			} else {
				mask >>= 1;
			}

			//TODO quitar mensaje
			printf(" %f ", results[i]);
			if (results[i] > 0){
				vectorData[i/BITS_PER_UNSIGNED] |= mask;
			} else {
				vectorData[i/BITS_PER_UNSIGNED] &= ~mask;
			}
		}
		//TODO quitar mensaje
		printf("\n");
	}
	mi_free(results);
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
