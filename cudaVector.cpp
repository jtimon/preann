/*
 * cudaVector.cpp
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#include "cudaVector.h"

CudaVector::CudaVector(unsigned size, VectorType vectorType, FunctionType functionType)
{
	this->size = size;
	this->vectorType = vectorType;
	switch (functionType){
		case FLOAT:
			this->functionType = functionType;
			break;
		case BIT:
			this->functionType = BINARY_STEP;
			break;
		case SIGN:
			this->functionType = BIPOLAR_STEP;
	}
	unsigned byte_sz = getByteSize();
	data = cuda_malloc(byte_sz);

	cuda_setZero(data, byte_sz, vectorType, CUDA_THREADS_PER_BLOCK);
}

CudaVector::~CudaVector()
{
	if (data) {
		cuda_free(data);
		data = NULL;
	}
}

void CudaVector::copyFrom(Interface *interface)
{
	if (size < interface->getSize()){
		string error = "The Interface is greater than the Vector.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
		string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}
	cuda_copyToDevice(data, interface->getDataPointer(), interface->getByteSize());
}

void CudaVector::copyTo(Interface *interface)
{
	if (interface->getSize() < size){
		string error = "The Vector is greater than the Interface.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
		string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}
	cuda_copyToHost(interface->getDataPointer(), data, this->getByteSize());
}

void CudaVector::activation(float* results)
{
	cuda_activation(data, size, vectorType, results, functionType, CUDA_THREADS_PER_BLOCK);
}

