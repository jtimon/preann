/*
 * cudaConnection.cpp
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#include "cudaConnection.h"

CudaConnection::CudaConnection(Vector* input, unsigned outputSize, VectorType vectorType): CudaVector(input->getSize() * outputSize, vectorType)
{
	tInput = input;
}

//TODO D igual que en la version Vector
void CudaConnection::crossover(Connection* other, Interface* bitVector)
{
	if (size != other->getSize()){
		std::string error = "The Connections must have the same size to crossover them.";
		throw error;
	}
	if (vectorType != other->getVectorType()){
		std::string error = "The Connections must have the same type to crossover them.";
		throw error;
	}
    CudaVector* cudaBitVector = new CudaVector(size, BIT, Cuda_Threads_Per_Block);
    cudaBitVector->copyFrom2(bitVector, Cuda_Threads_Per_Block);
    unsigned* cudaBitVectorPtr = (unsigned*)(cudaBitVector->getDataPointer());

    cuda_crossover(this->getDataPointer(), other->getDataPointer(), cudaBitVectorPtr, size, vectorType, Cuda_Threads_Per_Block);
    delete(cudaBitVector);
}

void CudaConnection::addToResults(Vector* results)
{
	void* inputWeighs = this->getDataPointer();
	float* resultsPtr = (float*)results->getDataPointer();
	// TODO TCC este método no funciona correctamente para SIGN
	cuda_inputCalculation(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
}
