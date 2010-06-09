/*
 * cudaLayer2.h
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#ifndef CUDALAYER_H_
#define CUDALAYER_H_

#include "layer.h"
#include "cudaVector.h"

class CudaLayer: public Layer {
protected:
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results);
	virtual float* negativeThresholds();

	virtual void init(unsigned size, VectorType outputType, FunctionType functionType);
	virtual void* newWeighs(unsigned inputSize, VectorType inputType);
	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);

	virtual void mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation);
	virtual void mutateThreshold(unsigned outputPos, float mutation);
public:
	static unsigned algorithm;
	static unsigned blockSize;
	CudaLayer();
	CudaLayer(unsigned size, VectorType outputType, FunctionType functionType);
	virtual ~CudaLayer();
	virtual ImplementationType getImplementationType() {
		return CUDA;
	};

	virtual void copyWeighs(Layer* sourceLayer);
	virtual void randomWeighs(float range);
	virtual void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector);

};

#endif /* CUDALAYER_H_ */
