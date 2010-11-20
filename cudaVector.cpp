/*
 * cudaVector.cpp
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#include "cudaVector.h"

//special constructor for bit coalescing vectors
CudaVector::CudaVector(Interface* bitVector, unsigned block_size)
{
	if (bitVector->getVectorType() != BIT){
		std::string error = "The Vector type must be BIT to use a BitVector CudaVector constructor.";
		throw error;
	}
	unsigned bitVectorSize = bitVector->getSize();
	unsigned maxWeighsPerBlock = BITS_PER_UNSIGNED * block_size;

	tSize = (bitVectorSize / maxWeighsPerBlock) * maxWeighsPerBlock;
	tSize += min(bitVectorSize % maxWeighsPerBlock, block_size) * BITS_PER_UNSIGNED;

	Interface interfaceOrderedByBlockSize = Interface(tSize, BIT);
	unsigned byteSize = interfaceOrderedByBlockSize.getByteSize();
	data = cuda_malloc(byteSize);

	unsigned bit = 0, thread = 0, block_offset = 0;
	for (unsigned i=0; i < bitVectorSize; i++){

		unsigned weighPos = (thread * BITS_PER_UNSIGNED) + bit + block_offset;
		thread++;
		interfaceOrderedByBlockSize.setElement(weighPos, bitVector->getElement(i));

		if (thread == block_size){
			thread = 0;
			bit++;
			if (bit == BITS_PER_UNSIGNED){
				bit = 0;
				block_offset += (block_size * BITS_PER_UNSIGNED);
			}
		}
	}
	cuda_copyToDevice(data, interfaceOrderedByBlockSize.getDataPointer(), byteSize);
}

CudaVector::CudaVector(unsigned size, VectorType vectorType)
{
	this->tSize = size;
	this->vectorType = vectorType;

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

Vector* CudaVector::clone()
{
	Vector* clone = new CudaVector(tSize, vectorType);
	copyTo(clone);
	return clone;
}

void CudaVector::copyFromImpl(Interface *interface)
{
	cuda_copyToDevice(data, interface->getDataPointer(), interface->getByteSize());
}

void CudaVector::copyToImpl(Interface *interface)
{
	cuda_copyToHost(interface->getDataPointer(), data, this->getByteSize());
}

void CudaVector::activation(Vector* resultsVect, FunctionType functionType)
{
	float* results = (float*)resultsVect->getDataPointer();
	cuda_activation(data, tSize, vectorType, results, functionType, CUDA_THREADS_PER_BLOCK);
}

void CudaVector::mutateImpl(unsigned pos, float mutation)
{
	cuda_mutate(data, pos, mutation, vectorType);
}

void CudaVector::crossoverImpl(Vector* other, Interface* bitVector)
{
    CudaVector* cudaBitVector = new CudaVector(bitVector, Cuda_Threads_Per_Block);

    cuda_crossover(this->getDataPointer(), other->getDataPointer(), (unsigned*)(cudaBitVector->getDataPointer()),
						tSize, vectorType, Cuda_Threads_Per_Block);
    delete(cudaBitVector);
}

unsigned CudaVector::getByteSize()
{
	switch (vectorType){
	case BYTE:
		return tSize;
		break;
	case FLOAT:
		return tSize * sizeof(float);
	case BIT:
	case SIGN:
		return (((tSize-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	}
}
