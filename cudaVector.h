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

	virtual unsigned getByteSize();
public:
	CudaVector() {};
	CudaVector(unsigned size, VectorType vectorType, unsigned block_size);
	CudaVector(unsigned size, VectorType vectorType);
	virtual ~CudaVector();

	//TODO deshacerse de CudaVector::copyFrom2
	virtual void copyFrom2(Interface* interface, unsigned block_size);

	virtual Vector* clone();
	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(Vector* results, FunctionType functionType);
	virtual void mutate(unsigned pos, float mutation);
};

#endif /* CUDAVECTOR_H_ */
