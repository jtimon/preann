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
public:
	CppLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~CppLayer();

	//void addInput(Vector* input);
	virtual void setSizes(unsigned totalWeighsPerOutput, unsigned outputSize);
	virtual void calculateOutput();
	virtual Layer* newCopy();

	virtual void randomWeighs(float range);
};

#endif /* CPPLAYER_H_ */
