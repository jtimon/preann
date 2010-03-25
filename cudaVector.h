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
	virtual void free();
public:
	CudaVector(unsigned size, VectorType vectorType);
	virtual ~CudaVector();

	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(float* results, FunctionType functionType);
};

#endif /* CUDAVECTOR_H_ */
