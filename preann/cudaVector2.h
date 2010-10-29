/*
 * CudaVector2.h
 *
 *  Created on: Oct 29, 2010
 *      Author: timon
 */

#ifndef CUDAVECTOR2_H_
#define CUDAVECTOR2_H_

#include "cudaVector.h"

class CudaVector2: public CudaVector {

public:
	CudaVector2(){}
	CudaVector2(unsigned size, VectorType vectorType, unsigned block_size);
	CudaVector2(unsigned size, VectorType vectorType);
	virtual ~CudaVector2();

	Vector* clone();
	virtual void inputCalculation(Vector* input, Vector* inputWeighs);
	//for weighs
	virtual void mutate(unsigned pos, float mutation, unsigned inputSize);
	virtual void weighCrossover(Vector* other, Vector* bitVector, unsigned inputSize);
};

#endif /* CUDAVECTOR2_H_ */
