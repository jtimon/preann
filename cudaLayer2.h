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
protected:
	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);
public:
	static unsigned algorithm;
	static unsigned blockSize;
	CudaLayer2(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~CudaLayer2();

	virtual void calculateOutput();
	void setSizes(unsigned  totalWeighsPerOutput, unsigned  outputSize);
	virtual Vector* newVector(unsigned size, VectorType vectorType);
	virtual Layer* newCopy();

	virtual void randomWeighs(float range);

};

#endif /* CUDALAYER2_H_ */
