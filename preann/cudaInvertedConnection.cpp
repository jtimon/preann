/*
 * cudaInvertedConnection.cpp
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#include "cudaInvertedConnection.h"

CudaInvertedConnection::CudaInvertedConnection(Vector* input, unsigned outputSize, VectorType vectorType): CudaVector(input->getSize() * outputSize, vectorType)
{
	tInput = input;
}

void CudaInvertedConnection::mutate(unsigned pos, float mutation)
{
	if (pos > size){
		std::string error = "The position being mutated is greater than the size of the vector.";
		throw error;
	}
	//TODO z simplificar cuentas
	unsigned outputPos = pos / tInput->getSize();
	unsigned inputPos = (pos % tInput->getSize());
	unsigned outputSize = size / tInput->getSize();
	pos = outputPos + (inputPos * outputSize);

	cuda_mutate(data, pos, mutation, vectorType);
}

void CudaInvertedConnection::crossoverImpl(Connection* other, Interface* bitVector)
{
	Interface* invertedBitVector = new Interface(size, BIT);

	unsigned width = size / tInput->getSize();
	unsigned height = tInput->getSize();

	for (unsigned i=0; i < width; i++){
		for (unsigned j=0; j < height; j++){
			invertedBitVector->setElement(i  + (j * width), bitVector->getElement((i * height) + j));
		}
	}
	CudaVector* cudaBitVector = new CudaVector(size, BIT, Cuda_Threads_Per_Block);
	cudaBitVector->copyFrom2(bitVector, Cuda_Threads_Per_Block);
	unsigned* cudaBitVectorPtr = (unsigned*)(cudaBitVector->getDataPointer());

	cuda_crossover(this->getDataPointer(), other->getDataPointer(), cudaBitVectorPtr, size, vectorType, Cuda_Threads_Per_Block);

	delete(cudaBitVector);
	delete(invertedBitVector);
}

void CudaInvertedConnection::addToResults(Vector* results)
{
	void* inputWeighs = this->getDataPointer();
	float* resultsPtr = (float*)results->getDataPointer();

	cuda_inputCalculationInvertedMatrix(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
}

void CudaInvertedConnection::copyFromImpl(Interface* interface)
{
	interface->transposeMatrix(tInput->getSize());
	CudaVector::copyFromImpl(interface);
}

void CudaInvertedConnection::copyToImpl(Interface* interface)
{
	unsigned outputSize = size / tInput->getSize();
	CudaVector::copyToImpl(interface);
	interface->transposeMatrix(outputSize);
}

//void CudaInvertedConnection::copyFromImpl(Interface* interface)
//{
//	unsigned outputSize = size / tInput->getSize();
//	interface->transposeMatrix(outputSize);
//
//	CudaVector::copyFromImpl(interface);
//}
//
//void CudaInvertedConnection::copyToImpl(Interface* interface)
//{
//	interface->transposeMatrix(tInput->getSize());
//	CudaVector::copyToImpl(interface);
//}
