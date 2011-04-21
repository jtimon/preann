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

template <VectorType vectorTypeTempl, class c_typeTempl>
class CudaConnection: public virtual Connection, public CudaVector<vectorTypeTempl, c_typeTempl> {
protected:
	virtual void mutateImpl(unsigned pos, float mutation)
	{
		cuda_mutate(data, pos, mutation, vectorTypeTempl);
	}

	virtual void crossoverImpl(Vector* other, Interface* bitVector)
	{
		CudaVector<vectorTypeTempl, c_typeTempl> cudaBitVector(bitVector, Cuda_Threads_Per_Block);

	    cuda_crossover(this->getDataPointer(), other->getDataPointer(), (unsigned*)cudaBitVector.getDataPointer(),
							tSize, vectorTypeTempl, Cuda_Threads_Per_Block);
	}
public:
	CudaConnection(Vector* input, unsigned outputSize)
		: CudaVector<vectorTypeTempl, c_typeTempl>(input->getSize() * outputSize)
	{
		tInput = input;
	}

	virtual ~CudaConnection() {};

	virtual void calculateAndAddTo(Vector* results)
	{
		void* inputWeighs = this->getDataPointer();
		float* resultsPtr = (float*)results->getDataPointer();
		// TODO TCC este método no funciona correctamente para SIGN
		cuda_inputCalculation(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
	}

};



#endif /* CUDACONNECTION_H_ */
