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

class CudaVector: virtual public Vector {
protected:
	unsigned getByteSize();
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
public:
	static unsigned algorithm;
	CudaVector() {};
	CudaVector(unsigned size, VectorType vectorType, unsigned block_size);
	CudaVector(unsigned size, VectorType vectorType);
	virtual ~CudaVector();
	virtual ImplementationType getImplementationType() {
		return CUDA;
	};

	//TODO D deshacerse de CudaVector::copyFrom2 o ponerle otro nombre
	virtual void copyFrom2(Interface* interface, unsigned block_size);

	virtual Vector* clone();
	virtual void activation(Vector* results, FunctionType functionType);

	virtual void mutate(unsigned pos, float mutation);
	virtual void crossover(Vector* other, Interface* bitVector);
};

#endif /* CUDAVECTOR_H_ */
