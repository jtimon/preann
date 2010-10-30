/*
 * cudaVector.h
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#ifndef CUDAVECTOR_H_
#define CUDAVECTOR_H_

#include "vector.h"
#include "cuda_code.h"

class CudaVector: public Vector {
protected:
	virtual unsigned getByteSize();
public:
	static unsigned algorithm;
	CudaVector() {};
	CudaVector(unsigned size, VectorType vectorType, unsigned block_size);
	CudaVector(unsigned size, VectorType vectorType);
	virtual ~CudaVector();
	virtual ImplementationType getImplementationType() {
		return CUDA;
	};

	//TODO deshacerse de CudaVector::copyFrom2
	virtual void copyFrom2(Interface* interface, unsigned block_size);

	virtual Vector* clone();
	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void inputCalculation(Vector* input, Vector* inputWeighs);
	virtual void activation(Vector* results, FunctionType functionType);
	virtual void mutate(unsigned pos, float mutation);
	//for weighs
	virtual void mutate(unsigned pos, float mutation, unsigned inputSize);
	virtual void weighCrossover(Vector* other, Interface* bitVector, unsigned inputSize);
};

#endif /* CUDAVECTOR_H_ */
