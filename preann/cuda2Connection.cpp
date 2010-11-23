/*
 * cuda2Connection.cpp
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#include "cuda2Connection.h"

Cuda2Connection::Cuda2Connection(Vector* input, unsigned outputSize, VectorType vectorType) : CudaConnection(input, outputSize, vectorType)
{
}

void Cuda2Connection::calculateAndAddTo(Vector* results)
{
	void* inputWeighs = this->getDataPointer();
	float* resultsPtr = (float*)results->getDataPointer();
	// TODO TCC este método no funciona correctamente para SIGN
	cuda_inputCalculationReduction(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
}
