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
	//TODO W!!! cuidadin que ya nadie usa esto
	virtual ImplementationType getImplementationType() {
		return CUDA2;
	};

	Vector* clone();
	//for weighs
	virtual void inputCalculation(Vector* results, Vector* input);
	virtual void mutate(unsigned pos, float mutation);
	virtual void weighCrossover(Vector* other, Interface* bitVector);

	unsigned char requiresTransposing(){
		return 1;
	}
};

#endif /* CUDAVECTOR2_H_ */
