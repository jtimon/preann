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

void CudaInvertedConnection::mutateImpl(unsigned pos, float mutation)
{
	//TODO z simplificar cuentas
	unsigned outputPos = pos / tInput->getSize();
	unsigned inputPos = (pos % tInput->getSize());
	unsigned outputSize = tSize / tInput->getSize();
	pos = outputPos + (inputPos * outputSize);

	cuda_mutate(data, pos, mutation, vectorType);
}

void CudaInvertedConnection::crossoverImpl(Vector* other, Interface* bitVector)
{
	Interface invertedBitVector = Interface(bitVector);
	invertedBitVector.transposeMatrix(tInput->getSize());

	CudaVector::crossoverImpl(other, &invertedBitVector);
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
	CudaVector::copyToImpl(interface);
	interface->transposeMatrix(tSize / tInput->getSize());
}
