#include "xmmNeuralNet.h"

XmmNeuralNet::XmmNeuralNet():NeuralNet()
{
}

XmmNeuralNet::XmmNeuralNet(unsigned  maxInputs, unsigned  maxLayers, unsigned  maxOutputs):NeuralNet(maxInputs, maxLayers, maxOutputs)
{
}

Layer* XmmNeuralNet::newLayer()
{
	return new XmmLayer();
}

Layer* XmmNeuralNet::newLayer(VectorType inputType, VectorType outputType, FunctionType functionType)
{
	return new XmmLayer(inputType, outputType, functionType);
}

Vector* XmmNeuralNet::newVector(unsigned size, VectorType vectorType)
{
	return new XmmVector(size, vectorType);
}

XmmNeuralNet::~XmmNeuralNet()
{
}
