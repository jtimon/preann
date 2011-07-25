
#include "inputLayer.h"
#include "factory.h"

InputLayer::InputLayer(unsigned size, VectorType vectorType, ImplementationType implementationType)
{
	tInput = new Interface(size, vectorType);
	output = Factory::newVector(size, vectorType, implementationType);

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

void InputLayer::addInput(Vector *input)
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


