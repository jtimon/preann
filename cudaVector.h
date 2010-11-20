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
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);
	CudaVector(Interface* bitVector, unsigned block_size);
public:
	CudaVector() {};
	CudaVector(unsigned size, VectorType vectorType);
	virtual ~CudaVector();
	virtual ImplementationType getImplementationType() {
		return CUDA;
	};

	virtual Vector* clone();
	virtual void activation(Vector* results, FunctionType functionType);
};

#endif /* CUDAVECTOR_H_ */
