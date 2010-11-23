/*
 * cudaConnection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDACONNECTION_H_
#define CUDACONNECTION_H_

#include "connection.h"
#include "cudaVector.h"

template <VectorType vectorTypeTempl>
class CudaConnection: public virtual Connection, public CudaVector<vectorTypeTempl> {
public:
	CudaConnection(Vector* input, unsigned outputSize);
	virtual ~CudaConnection() {};

	virtual void calculateAndAddTo(Vector* results);
};

template <VectorType vectorTypeTempl>
CudaConnection<vectorTypeTempl>::CudaConnection(Vector* input, unsigned outputSize)
	: CudaVector<vectorTypeTempl>(input->getSize() * outputSize)
{
	tInput = input;
}

template <VectorType vectorTypeTempl>
void CudaConnection<vectorTypeTempl>::calculateAndAddTo(Vector* results)
{
	void* inputWeighs = this->getDataPointer();
	float* resultsPtr = (float*)results->getDataPointer();
	// TODO TCC este método no funciona correctamente para SIGN
	cuda_inputCalculation(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
}

#endif /* CUDACONNECTION_H_ */
