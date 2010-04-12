/*
 * cudaVector.cpp
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#include "cudaVector.h"

CudaVector::CudaVector(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;
	unsigned byte_sz = getByteSize();

	cudaMalloc((void**)&(data), byte_sz);
	checkCUDAError("CudaVector::CudaVector");

	cuda_setZero(data, byte_sz, vectorType, THREADS_PER_BLOCK);
}

CudaVector::~CudaVector()
{
	if (data) {
		cudaFree(data);
		checkCUDAError("CudaVector::~CudaVector()");
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
	cudaMemcpy(data, interface->getDataPointer(), interface->getByteSize(), cudaMemcpyHostToDevice);
	checkCUDAError("CudaVector::copyFrom");
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
	cudaMemcpy(interface->getDataPointer(), data, this->getByteSize(), cudaMemcpyDeviceToHost);
	checkCUDAError("CudaVector::copyTo");
}

void CudaVector::activation(float* results, FunctionType functionType)
{
	cuda_activation(data, size, vectorType, results, functionType, THREADS_PER_BLOCK);
}

