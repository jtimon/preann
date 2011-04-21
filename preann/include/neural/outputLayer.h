/*
 * outputLayer.h
 *
 *  Created on: Dec 4, 2010
 *      Author: timon
 */

#ifndef OUTPUTLAYER_H_
#define OUTPUTLAYER_H_

#include "layer.h"

class OutputLayer: public Layer {
protected:
	Interface* tOuputInterface;
	OutputLayer() {};
public:
	OutputLayer(unsigned size, VectorType outputType, FunctionType functionType, ImplementationType implementationType);
	virtual ~OutputLayer();

	void calculateOutput();
	Interface* getOutputInterface();
};

#endif /* OUTPUTLAYER_H_ */
