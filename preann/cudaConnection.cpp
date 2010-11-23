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

void CudaConnection::calculateAndAddTo(Vector* results)
{
	void* inputWeighs = this->getDataPointer();
	float* resultsPtr = (float*)results->getDataPointer();
	// TODO TCC este método no funciona correctamente para SIGN
	cuda_inputCalculation(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
}
