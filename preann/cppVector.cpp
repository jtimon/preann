/*
 * cppVector.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "cppVector.h"

CppVector::CppVector(unsigned size, VectorType vectorType, FunctionType functionType)
{
	this->size = size;
	this->vectorType = vectorType;
	this->functionType = functionType;

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	if (vectorType == FLOAT){

		unsigned floatSize = byteSize/sizeof(float);
		for (unsigned i=0; i< floatSize; i++){
			((float*)data)[i] = 0;
		}
	}
	else {
		for (unsigned i=0; i < byteSize; i++){
			((unsigned char*)data)[i] = 0;
		}
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

void CppVector::activation(float* results)
{
	if (vectorType == FLOAT){
		for (unsigned i=0; i < size; i++){
			((float*)data)[i] = Function(results[i], functionType);
		}
	} else {
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
	mi_free(results);
}

unsigned CppVector::getByteSize()
{
	if (vectorType == FLOAT){

		return size * sizeof(float);
	}
	else {
		return (((size-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	}
}
