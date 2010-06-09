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
public:
	CudaVector(unsigned size, VectorType vectorType, FunctionType functionType);
	virtual ~CudaVector();

	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(float* results);
};

#endif /* CUDAVECTOR_H_ */
