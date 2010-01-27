#include "xmmLayer.h"

XmmLayer::XmmLayer()
{
}

XmmLayer::XmmLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
}

XmmLayer::~XmmLayer()
{
}

Vector* XmmLayer::newVector(unsigned size, VectorType vectorType)
{
	return new XmmVector(size, vectorType);
}

void XmmLayer::calculateOutput()
{
	float result;
	unsigned w = 0;
	switch (inputType) {
		case FLOAT:
			for (unsigned i=0; i < output->getSize(); i++){
				result = 0;
				for (unsigned j=0; j < numberInputs; j++){
					float auxResult;
					XMMreal(inputs[j]->getDataPointer(), ((XmmVector*)inputs[j])->getNumLoops(), ((float*)weighs + w), auxResult);
					result += auxResult;
					w += inputs[j]->getWeighsSize();
				}
				output->setElement(i, Function(result - thresholds[i], functionType));
			}
			break;
		case BIT:
			for (unsigned i=0; i < output->getSize(); i++){
				result = 0;
				for (unsigned j=0; j < numberInputs; j++){
					result += XMMbinario(inputs[j]->getDataPointer(), ((XmmVector*)inputs[j])->getNumLoops(), ((unsigned char*)weighs + w));
					w += inputs[j]->getWeighsSize();
				}
				output->setElement(i, Function(result - thresholds[i], functionType));
			}
			break;
		case SIGN:
			for (unsigned i=0; i < output->getSize(); i++){
				result = 0;
				for (unsigned j=0; j < numberInputs; j++){
					result += XMMbipolar(inputs[j]->getDataPointer(), ((XmmVector*)inputs[j])->getNumLoops(), ((unsigned char*)weighs + w));
					w += inputs[j]->getWeighsSize();
				}
				output->setElement(i, Function(result - thresholds[i], functionType));
			}
			break;
	}
}
