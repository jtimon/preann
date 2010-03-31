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
	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);
	virtual float* negativeThresholds();
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results);
	virtual void* newWeighs(unsigned inputSize, VectorType inputType);
public:
	CppLayer(VectorType outputType, FunctionType functionType);
	CppLayer(unsigned size, VectorType outputType, FunctionType functionType);
	virtual ~CppLayer();

	virtual void randomWeighs(float range);

	virtual Layer* newCopy();
};

#endif /* CPPLAYER_H_ */
