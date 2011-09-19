
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

void InputLayer::addInput(Buffer *input)
{
	std::string error = "addInput method does not work for InputLayer.";
	throw error;
}

void InputLayer::calculateOutput()
{
	output->copyFromInterface(tInput);
}

Interface* InputLayer::getInputInterface()
{
	return tInput;
}


