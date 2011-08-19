
#include "inputLayer.h"
#include "factory.h"

InputLayer::InputLayer(unsigned size, BufferType bufferType, ImplementationType implementationType)
{
	tInput = new Interface(size, bufferType);
	output = Factory::newBuffer(size, bufferType, implementationType);

	thresholds = NULL;
	functionType = IDENTITY;

	connections = NULL;
	numberInputs = 0;
}

InputLayer::~InputLayer()
{
	if (tInput){
		delete(tInput);
	}
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


