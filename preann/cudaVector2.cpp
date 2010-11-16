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
	Vector* clone = new CudaVector2(size, vectorType);
	copyToVector(clone);
	return clone;
}

void CudaVector2::mutate(unsigned pos, float mutation)
{
	//TODO W eliminar
}
void CudaVector2::weighCrossover(Vector* other, Interface* bitVector)
{
	//TODO W eliminar
}
