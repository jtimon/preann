/*
 * inputLayer.h
 *
 *  Created on: Nov 28, 2010
 *      Author: timon
 */

#ifndef INPUTLAYER_H_
#define INPUTLAYER_H_

#include "layer.h"

class InputLayer: public Layer {
	Interface* tInput;
protected:
	ImplementationType getImplementationType();
public:
	InputLayer(unsigned size, VectorType vectorType, ImplementationType implementationType);
	virtual ~InputLayer();

	void addInput(Vector* input);
	void calculateOutput();

	Interface* getInputInterface();
};

#endif /* INPUTLAYER_H_ */
