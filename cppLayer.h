/*
 * cppLayer.h
 *
 *  Created on: Mar 26, 2010
 *      Author: timon
 */

#ifndef CPPLAYER_H_
#define CPPLAYER_H_

#include "layer.h"

class CppLayer : public Layer
{
protected:
	virtual float* negativeThresholds();
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results);

	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);
	virtual void* newWeighs(unsigned inputSize, VectorType inputType);

	virtual void mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation);
	virtual void mutateThreshold(unsigned outputPos, float mutation);

	CppLayer();
public:
	CppLayer(unsigned size, VectorType outputType, FunctionType functionType);
	virtual ~CppLayer();
	virtual ImplementationType getImplementationType() {
		return C;
	};

	virtual void copyWeighs(Layer* sourceLayer);
	virtual void randomWeighs(float range);
	virtual void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector);

};

#endif /* CPPLAYER_H_ */
