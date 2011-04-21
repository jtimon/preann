/*
 * outputLayer.cpp
 *
 *  Created on: Dec 4, 2010
 *      Author: timon
 */

#include "outputLayer.h"

OutputLayer::OutputLayer(unsigned size, VectorType outputType,
		FunctionType functionType, ImplementationType implementationType) :
	Layer(size, outputType, functionType, implementationType) {

	tOuputInterface = new Interface(size, outputType);
}

OutputLayer::~OutputLayer() {
	delete (tOuputInterface);
}

void OutputLayer::calculateOutput() {
	Layer::calculateOutput();
	output->copyToInterface(tOuputInterface);
}

Interface *OutputLayer::getOutputInterface()
{
	return tOuputInterface;
}



