/*
 * cudaLayer2.h
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#ifndef CUDALAYER2_H_
#define CUDALAYER2_H_

#include "layer.h"
#include "cudaVector.h"

class CudaLayer2: public Layer {
public:
	CudaLayer2(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~CudaLayer2();

	virtual void calculateOutput();

	virtual Vector* newVector(unsigned size, VectorType vectorType);
	virtual Layer* newCopy();

	virtual void randomWeighs(float range);
	virtual void save(FILE* stream);
	virtual void load(FILE* stream);
};

#endif /* CUDALAYER2_H_ */
