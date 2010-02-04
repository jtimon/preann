#ifndef XMMNEURALNET_H_
#define XMMNEURALNET_H_

#include "neuralNet.h"
#include "xmmLayer.h"

class XmmNeuralNet: public NeuralNet {
protected:
	//using NeuralNet::addLayer;
public:
	XmmNeuralNet();
	virtual ~XmmNeuralNet();

	virtual Layer* newLayer();
	virtual Layer* newLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual Vector* newVector(unsigned size, VectorType vectorType);
};

#endif /* XMMNEURALNET_H_ */
