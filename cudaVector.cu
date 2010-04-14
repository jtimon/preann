/*
 * cudaVector.cpp
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#include "cudaVector.h"

__global__
void setZeroKernel(float* data, unsigned size)
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

	if (vectorType == FLOAT){
		unsigned size = byte_sz/sizeof(float);
		unsigned grid_size = ((size - 1)/THREADS_PER_BLOCK) + 1;
		setZeroKernel<<< grid_size, THREADS_PER_BLOCK >>>((float*)data, size);
	} else {
		cudaMemset(data, 0, byte_sz);
	}
	checkCUDAError("CudaVector::CudaVector");
}

CudaVector::~CudaVector()
{
	if (data) {
		cudaFree(data);
		data = NULL;
	}
	checkCUDAError("CudaVector::~CudaVector()");
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

__device__
float Func(float number, FunctionType functionType) {

	switch (functionType) {

		//TODO añadir diferentes funciones

		case BINARY_STEP:
			if (number > 0){
				return 1;
			} else {
				return 0;
			}
		case BIPOLAR_STEP:
			if (number > 0){
				return 1;
			} else {
				return -1;
			}
		//case ANOTHER_FUNCTION:
		//	return anotherFunction(number);

		case IDENTITY:
		default:
			return number;
	}
}

__global__
void activation_float_kernel(float* results, float* output, unsigned output_sz, FunctionType functionType)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < output_sz) output[idx] = Func(results[idx], functionType);
}

__global__
void activation_bit_kernel(float* results, unsigned* output, unsigned output_sz)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned offset = idx * BITS_PER_UNSIGNED;

	if (output_sz > offset){

		unsigned toRead = min(BITS_PER_UNSIGNED, output_sz - offset);
		unsigned threadOutput = 0;
		unsigned mask = 0x80000000;

		for (unsigned i=0; i < toRead; i++){
			if (results[offset + i] > 0){
				threadOutput |= mask;
			} else {
				threadOutput &= ~mask;
			}
			mask >>= 1;
		}
		output[idx] = threadOutput;
	}
}

void CudaVector::activation(float* results, FunctionType functionType)
{
	//cuda_activation(void* data, unsigned size, VectorType vectorType, float* results, FunctionType functionType, unsigned block_size){
	//cuda_activation(data, size, vectorType, results, functionType, THREADS_PER_BLOCK);
	unsigned grid_size;

	if (vectorType == FLOAT) {
		grid_size = ((size - 1)/THREADS_PER_BLOCK) + 1;
		activation_float_kernel<<< grid_size, THREADS_PER_BLOCK >>>(results, (float*)data, size, functionType);
	} else {
		grid_size = ((size - 1) / (THREADS_PER_BLOCK * BITS_PER_UNSIGNED)) + 1;
		activation_bit_kernel<<< grid_size, THREADS_PER_BLOCK >>>(results, (unsigned*)data, size);
	}
	cudaFree(results);
	checkCUDAError("CudaVector::activation");
}

