/*
 * CudaVector2.cpp
 *
 *  Created on: Oct 29, 2010
 *      Author: timon
 */

#include "cudaVector2.h"

CudaVector2::CudaVector2(unsigned size, VectorType vectorType, unsigned block_size)
{
	(((size-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	this->size = size;
	this->vectorType = vectorType;

	unsigned byte_sz = ((size-1)/(BITS_PER_UNSIGNED * block_size)+1) * (sizeof(unsigned) * block_size);
	data = cuda_malloc(byte_sz);

	cuda_setZero(data, byte_sz, vectorType, CUDA_THREADS_PER_BLOCK);
}

CudaVector2::CudaVector2(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

	unsigned byte_sz = getByteSize();
	data = cuda_malloc(byte_sz);

	cuda_setZero(data, byte_sz, vectorType, CUDA_THREADS_PER_BLOCK);
}

CudaVector2::~CudaVector2()
{
	if (data) {
		cuda_free(data);
		data = NULL;
	}
}

Vector* CudaVector2::clone()
{
	//TODO implementar CudaVector2::clone()
	Vector* clone = new CudaVector2(size, vectorType);
	copyToVector(clone);
	return clone;
}

void CudaVector2::inputCalculation(Vector* resultsVect, Vector* input)
{
	void* inputWeighs = this->getDataPointer();
	float* results = (float*)resultsVect->getDataPointer();

	cuda_inputCalculationInvertedMatrix(input->getDataPointer(), input->getSize(), input->getVectorType(), resultsVect->getSize(), inputWeighs, results, Cuda_Threads_Per_Block);
}

void CudaVector2::mutate(unsigned pos, float mutation)
{
	cuda_mutate(data, pos, mutation, vectorType);
}
void CudaVector2::weighCrossover(Vector* other, Interface* bitVector)
{
    CudaVector2* cudaBitVector = new CudaVector2(size, BIT, Cuda_Threads_Per_Block);
    cudaBitVector->copyFrom2(bitVector, Cuda_Threads_Per_Block);
    unsigned* cudaBitVectorPtr = (unsigned*)(cudaBitVector->getDataPointer());

    cuda_crossover(this->getDataPointer(), other->getDataPointer(), cudaBitVectorPtr, size, vectorType, Cuda_Threads_Per_Block);
    delete(cudaBitVector);
}
