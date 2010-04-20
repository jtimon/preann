/*
 * cudaLayer2.h
 *
 *  Created on: Apr 15, 2010
 *      Author: timon
 */

#ifndef CUDALAYER2_H_
#define CUDALAYER2_H_

#include "cudaLayer.h"

class CudaLayer2: public CudaLayer {
protected:
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results);

	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);

	virtual void mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation);

	void transposeMatrix(void* matrix, unsigned width, unsigned height, VectorType inputType);
public:
	CudaLayer2(unsigned size, VectorType outputType, FunctionType functionType);
	virtual ~CudaLayer2();
	virtual ImplementationType getImplementationType() {
		return CUDA2;
	};

	virtual void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector);
};

#endif /* CUDALAYER2_H_ */
