
#include "inputLayer.h"
#include "factory.h"

InputLayer::InputLayer(Interface* interface, ImplementationType implementationType)
{
	tInput = interface;
	output = Factory::newBuffer(interface->getSize(), interface->getBufferType(), implementationType);

	thresholds = NULL;
	functionType = IDENTITY;
}

InputLayer::InputLayer(unsigned size, BufferType bufferType, ImplementationType implementationType)
{
	//TODO quitar esto
	tInput = new Interface(size, bufferType);
	output = Factory::newBuffer(size, bufferType, implementationType);

	thresholds = NULL;
	functionType = IDENTITY;
}

InputLayer::~InputLayer()
{
	//TODO quitar esto
//	if (tInput){
//		delete(tInput);
//	}
}

ImplementationType InputLayer::getImplementationType()
{
	output->getImplementationType();
}

void InputLayer::addInput(Buffer *input)
{
	std::string error = "addInput method does not work for InputLayer.";
	throw error;
}

void InputLayer::calculateOutput()
{
	output->copyFromInterface(tInput);
}

void InputLayer::copyWeighs(Layer* sourceLayer)
{
	if(connections.size() != sourceLayer->getNumberInputs()){
		std::string error = "InputLayer::copyWeighs : Cannot copyWeighs from a layer with " +
			to_string(sourceLayer->getNumberInputs()) + " connections to a layer with " +
			to_string(connections.size());
		throw error;
	}
	if (this->getImplementationType() != sourceLayer->getImplementationType()){
		std::string error = "InputLayer::copyWeighs : The layers are incompatible: the implementation is different.";
		throw error;
	}
	if (sourceLayer->getThresholds() != NULL){
		std::string error = "InputLayer::copyWeighs : trying to copy from a non input layer";
		throw error;
	}
}

Interface* InputLayer::getInputInterface()
{
	return tInput;
}


