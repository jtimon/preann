#include "xmmLayer.h"

XmmLayer::XmmLayer()
{
	printf("se construye layer XMM\n");
}

XmmLayer::XmmLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
	printf("se construye layer XMM parametrizada\n");
}

XmmLayer::~XmmLayer()
{
}

Layer* XmmLayer::newCopy()
{
	Layer* copy = new XmmLayer(inputType, outputType, functionType);

	copy->setSizes(totalWeighsPerOutput, output->getSize());

	return copy;
}

Vector* XmmLayer::newVector(unsigned size, VectorType vectorType)
{
	return new XmmVector(size, vectorType);
}

void XmmLayer::calculateOutput()
{
	if (!output) {
		string error = "Cannot calculate the output of a Layer without output.";
		throw error;
	}

	float result;
	unsigned w = 0;
	switch (inputType) {
		case FLOAT:
			for (unsigned i=0; i < output->getSize(); i++){
				result = 0;
				for (unsigned j=0; j < numberInputs; j++){
					float auxResult;
					XMMreal(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((float*)weighs + w), auxResult);
					result += auxResult;
					w += getInput(j)->getWeighsSize();
				}
				result -= thresholds[i];
				output->setElement(i, Function(result, functionType));
			}
			break;
		case BIT:
			for (unsigned i=0; i < output->getSize(); i++){
				result = 0;
				for (unsigned j=0; j < numberInputs; j++){
					int auxResult = 0;
					XMMbinario(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + w), auxResult);
					result += auxResult;
					w += getInput(j)->getWeighsSize();
				}
				result -= thresholds[i];
				output->setElement(i, Function(result, functionType));
			}
			break;
		case SIGN:
			for (unsigned i=0; i < output->getSize(); i++){
				result = 0;
				for (unsigned j=0; j < numberInputs; j++){
					result += XMMbipolar(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + w));
					w += getInput(j)->getWeighsSize();
				}
				result -= thresholds[i];
				output->setElement(i, Function(result, functionType));
			}
			break;
	}
}
