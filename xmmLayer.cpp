#include "xmmLayer.h"
/*
XmmLayer::XmmLayer()
{
}*/

XmmLayer::XmmLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
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
	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	unsigned w = 0;

	switch (inputType) {
		case FLOAT:
			for (unsigned i=0; i < output->getSize(); i++){
				results[i] = 0;
				for (unsigned j=0; j < numberInputs; j++){
					float auxResult;
					XMMreal(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((float*)weighs + w), auxResult);
					results[i] += auxResult;
					w += getInput(j)->getWeighsSize();
				}
				results[i] -= thresholds[i];
			}
			break;
		case BIT:
			for (unsigned i=0; i < output->getSize(); i++){
				results[i] = 0;
				for (unsigned j=0; j < numberInputs; j++){
					int auxResult = 0;
					XMMbinario(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + w), auxResult);
					results[i] += auxResult;
					w += getInput(j)->getWeighsSize();
				}
				results[i] -= thresholds[i];
			}
			break;
		case SIGN:
			for (unsigned i=0; i < output->getSize(); i++){
				results[i] = 0;
				for (unsigned j=0; j < numberInputs; j++){
					results[i] += XMMbipolar(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + w));
					w += getInput(j)->getWeighsSize();
				}
				results[i] -= thresholds[i];
			}
			break;
	}

	 //TODO quitar getNumLoops (sacar el bucle de inputs del de salidas) y ¿permitir varios tipos de entrada?
/*	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	for (unsigned j=0; j < output->getSize(); j++) {
		results[j] = -thresholds[j];
	}

	unsigned inputOffset = 0;
	for (unsigned i=0; i < numberInputs; i++){

		void* input = getInput(i)->getDataPointer();

		for (unsigned j=0; j < output->getSize(); j++){
			unsigned weighPos = j*totalWeighsPerOutput + inputOffset;

				if (inputType == FLOAT) {
					float auxResult;
					XMMreal(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((float*)weighs + weighPos), auxResult);
					results[i] += auxResult;
					//results[j] += ((float*)input)[k] * ((float*)weighs)[weighPos];
				}
				if (inputType == BIT) {
					int auxResult = 0;
					XMMbinario(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + weighPos), auxResult);
					results[i] += auxResult;
//					if ( ((unsigned*)input)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
//						results[j] += (((unsigned char*)weighs)[weighPos] - 128);
//					}
				}
				if (inputType == SIGN) {
					results[i] += XMMbipolar(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + weighPos));
//					if ( ((unsigned*)input)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
//						results[j] += (((unsigned char*)weighs)[weighPos] - 128);
//					} else {
//						results[j] -= (((unsigned char*)weighs)[weighPos] - 128);
//					}
				}
		}
		inputOffset += getInput(i)->getWeighsSize();
	}*/

	printf("----------------\n", 1);
	for (unsigned i=0; i < output->getSize(); i++){
		printf("%f ", results[i]);
	}
	printf("\n----------------\n", 1);
	output->activation(results, functionType);
	mi_free(results);
}
