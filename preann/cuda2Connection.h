/*
 * cuda2Connection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDA2CONNECTION_H_
#define CUDA2CONNECTION_H_

#include "cudaConnection.h"

template <VectorType vectorTypeTempl>
class Cuda2Connection: virtual public Connection, public CudaConnection<vectorTypeTempl> {
public:
	Cuda2Connection(Vector* input, unsigned outputSize);
	virtual ~Cuda2Connection() {};
	virtual ImplementationType getImplementationType() {
		return CUDA2;
	};

	virtual void calculateAndAddTo(Vector* results);
};

template <VectorType vectorTypeTempl>
Cuda2Connection<vectorTypeTempl>::Cuda2Connection(Vector* input, unsigned outputSize)
		: CudaConnection<vectorTypeTempl>(input, outputSize)
{
}

template <VectorType vectorTypeTempl>
void Cuda2Connection<vectorTypeTempl>::calculateAndAddTo(Vector* results)
{
	void* inputWeighs = this->getDataPointer();
	float* resultsPtr = (float*)results->getDataPointer();
	// TODO TCC este método no funciona correctamente para SIGN
	cuda_inputCalculationReduction(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
}

#endif /* CUDA2CONNECTION_H_ */
