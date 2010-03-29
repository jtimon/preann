#include "xmmLayer.h"

XmmLayer::XmmLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
}

XmmLayer::~XmmLayer()
{
	if (inputs) {
		mi_free(inputs);
	}
	if (thresholds) {
		mi_free(thresholds);
	}
	if (weighs) {
		mi_free(weighs);
	}
	if (output) {
		delete (output);
	}
}

void XmmLayer::saveWeighs(FILE *stream)
{
	fwrite(thresholds, output->getSize() * sizeof(float), 1, stream);
	unsigned size;
	if (inputType == FLOAT){
		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
	} else {
		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
	}
	fwrite(weighs, size, 1, stream);
}

void XmmLayer::loadWeighs(FILE *stream)
{
	fread(thresholds, output->getSize() * sizeof(float), 1, stream);
	unsigned size;
	if (inputType == FLOAT){
		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
	} else {
		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
	}
	fread(weighs, size, 1, stream);
}

void XmmLayer::setSizes(unsigned  totalWeighsPerOutput, unsigned  outputSize)
{
	if (!output) {
		output = new XmmVector(outputSize, outputType);
		thresholds = (float*) mi_malloc(sizeof(float) * outputSize);
	} else if (output->getSize() != outputSize) {

		cout<<"Warning: a layer is changing the location of its output."<<endl;
		delete (output);
		if (thresholds) {
			mi_free(thresholds);
		}
		output = new XmmVector(outputSize, outputType);
		thresholds = (float*)mi_malloc(sizeof(float) * outputSize);
	}
	if (totalWeighsPerOutput > 0){
		if (inputType == FLOAT){

			weighs = mi_malloc(sizeof(float) * outputSize * totalWeighsPerOutput);
			for (unsigned i=0; i < outputSize * totalWeighsPerOutput; i++){
				((float*)weighs)[i] = 0;
			}
		} else {
			weighs = mi_malloc(sizeof(unsigned char) * outputSize * totalWeighsPerOutput);
			for (unsigned i=0; i < outputSize * totalWeighsPerOutput; i++){
				((unsigned char*)weighs)[i] = 128;
			}
		}
	}
	this->totalWeighsPerOutput = totalWeighsPerOutput;
}

Layer* XmmLayer::newCopy()
{
	Layer* copy = new XmmLayer(inputType, outputType, functionType);

	copy->setSizes(totalWeighsPerOutput, output->getSize());

	return copy;
}

void XmmLayer::calculateOutput()
{
	if (!output) {
		string error = "Cannot calculate the output of a Layer without output.";
		throw error;
	}/*
	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	unsigned w = 0;

	switch (inputType) {
		case FLOAT:
			for (unsigned i=0; i < output->getSize(); i++){
				results[i] = -thresholds[i];
				for (unsigned j=0; j < numberInputs; j++){
					float auxResult;
					getInput(j)->print();
					XMMreal(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((float*)weighs + w), auxResult);
					results[i] += auxResult;
					w += getInput(j)->getWeighsSize();
				}
			}
			break;
		case BIT:
			for (unsigned i=0; i < output->getSize(); i++){
				results[i] = -thresholds[i];
				for (unsigned j=0; j < numberInputs; j++){
					int auxResult = 0;
					XMMbinario(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + w), auxResult);
					results[i] += auxResult;
					w += getInput(j)->getWeighsSize();
				}
			}
			break;
		case SIGN:
			for (unsigned i=0; i < output->getSize(); i++){
				results[i] = -thresholds[i];
				for (unsigned j=0; j < numberInputs; j++){
					results[i] += XMMbipolar(getInput(j)->getDataPointer(), ((XmmVector*)getInput(j))->getNumLoops(), ((unsigned char*)weighs + w));
					w += getInput(j)->getWeighsSize();
				}
			}
			break;
	}*/

	 //TODO quitar getNumLoops (sacar el bucle de inputs del de salidas) y ¿permitir varios tipos de entrada?
	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	for (unsigned j=0; j < output->getSize(); j++) {
		results[j] = -thresholds[j];
	}

	unsigned inputOffset = 0;
	for (unsigned i=0; i < numberInputs; i++){

		void* inputPtr = getInput(i)->getDataPointer();

		for (unsigned j=0; j < output->getSize(); j++){
			unsigned weighPos = j*totalWeighsPerOutput + inputOffset;

				if (inputType == FLOAT) {
					float auxResult;
					XMMreal(inputPtr, ((XmmVector*)getInput(i))->getNumLoops(), ((float*)weighs + weighPos), auxResult);
					results[j] += auxResult;
					//results[j] += ((float*)input)[k] * ((float*)weighs)[weighPos];
				}
				if (inputType == BIT) {
					int auxResult = 0;
					XMMbinario(inputPtr, ((XmmVector*)getInput(i))->getNumLoops(), ((unsigned char*)weighs + weighPos), auxResult);
					results[j] += auxResult;
//					if ( ((unsigned*)input)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
//						results[j] += (((unsigned char*)weighs)[weighPos] - 128);
//					}
				}
				if (inputType == SIGN) {
					results[j] += XMMbipolar(inputPtr, ((XmmVector*)getInput(i))->getNumLoops(), ((unsigned char*)weighs + weighPos));
//					if ( ((unsigned*)input)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
//						results[j] += (((unsigned char*)weighs)[weighPos] - 128);
//					} else {
//						results[j] -= (((unsigned char*)weighs)[weighPos] - 128);
//					}
				}
		}
		inputOffset += getInput(i)->getWeighsSize();
	}

	printf("----------------\n", 1);
	for (unsigned i=0; i < output->getSize(); i++){
		printf("%f ", results[i]);
	}
	printf("\n----------------\n", 1);
	output->activation(results, functionType);
	mi_free(results);
}


