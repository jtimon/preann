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

class CudaLayer: public Layer {
protected:
	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);
	virtual float* negativeThresholds();
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results);
	virtual void* newWeighs(unsigned inputSize, VectorType inputType);
public:
	static unsigned algorithm;
	static unsigned blockSize;
	CudaLayer(unsigned size, VectorType outputType, FunctionType functionType);
	virtual ~CudaLayer();

	//virtual void calculateOutput();
	//virtual void addInput(Vector* input);
	virtual void randomWeighs(float range);

	virtual Layer* newCopy();


};

#endif /* CUDALAYER2_H_ */
