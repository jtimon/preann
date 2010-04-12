/*
 * cudaVector.cpp
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#include "cudaVector.h"

__global__ void setZeroKernel(float* data, unsigned size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) data[idx] = 0;
}

CudaVector::CudaVector(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;
	unsigned byte_sz = getByteSize();

	cudaMalloc((void**)&(data), byte_sz);
	checkCUDAError("CudaVector::CudaVector");

	//cuda_setZero(data, byte_sz, vectorType, THREADS_PER_BLOCK);
	if (vectorType == FLOAT){
		unsigned size = byte_sz/sizeof(float);
		unsigned grid_size = ((size - 1)/THREADS_PER_BLOCK) + 1;
		setZeroKernel<<< grid_size, THREADS_PER_BLOCK >>>((float*)data, size);
	} else {
		cudaMemset(data, 0, byte_sz);
	}
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

