/*
 * cppLayer.cpp
 *
 *  Created on: Mar 26, 2010
 *      Author: timon
 */

#include "cppLayer.h"

CppLayer::CppLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
}

CppLayer::~CppLayer()
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

void CppLayer::randomWeighs(float range)
{
	if (output == NULL){
		string error = "Cannot set random weighs to a layer with no output.";
		throw error;
	}
	if (numberInputs == 0){
		string error = "Cannot set random weighs to a layer with no inputs.";
		throw error;
	}
	if (inputType != FLOAT && range >= 128){
		range = 127;
	}
	for (unsigned i=0; i < output->getSize(); i++){

		//thresholds[i] = 0;
		thresholds[i] = randomFloat(range);
		unsigned inputOffset = 0;
		for (unsigned j=0; j < numberInputs; j++){

			unsigned inputSize = getInput(j)->getSize();
			for (unsigned k=0; k < inputSize; k++){

				unsigned weighPos = i*totalWeighsPerOutput + inputOffset + k;
				if (inputType == FLOAT) {
					//((float*)weighs)[weighPos] = 0;
					((float*)weighs)[weighPos] = randomFloat(range);
				} else {
					//TODO revisar el xmm a ver si se pueden usar char normales para los pesos (y no hacer el truco del 128)
					((unsigned char*)weighs)[weighPos] = 128 + (unsigned char)randomInt(range);
				}
			}
			inputOffset += getInput(j)->getWeighsSize();
		}
	}
}

void CppLayer::loadWeighs(FILE *stream)
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

void CppLayer::saveWeighs(FILE *stream)
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

Layer* CppLayer::newCopy()
{
	Layer* copy = new CppLayer(inputType, outputType, functionType);

	copy->setSizes(totalWeighsPerOutput, output->getSize());

	return copy;
}


void CppLayer::setSizes(unsigned  totalWeighsPerOutput, unsigned  outputSize)
{
	if (!output) {
		output = new Vector(outputSize, outputType);
		thresholds = (float*) mi_malloc(sizeof(float) * outputSize);
	} else if (output->getSize() != outputSize) {

		cout<<"Warning: a layer is changing the location of its output."<<endl;
		delete (output);
		if (thresholds) {
			mi_free(thresholds);
		}
		output = new Vector(outputSize, outputType);
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

void CppLayer::calculateOutput()
{
	if (!output) {
		string error = "Cannot calculate the output of a Layer without output.";
		throw error;
	}

	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	for (unsigned j=0; j < output->getSize(); j++) {
		results[j] = -thresholds[j];
	}

	unsigned inputOffset = 0;
	for (unsigned i=0; i < numberInputs; i++){

		void* input = getInput(i)->getDataPointer();

		for (unsigned j=0; j < output->getSize(); j++){

			for (unsigned k=0; k < getInput(i)->getSize(); k++){
				unsigned weighPos = j*totalWeighsPerOutput + inputOffset + k;
				if (inputType == FLOAT) {
					//printf("i % d input %f weigh %f \n", k, ((float*)input)[k], ((float*)weighs)[weighPos]);
					results[j] += ((float*)input)[k] * ((float*)weighs)[weighPos];
				} else {
					if ( ((unsigned*)input)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
						results[j] += (((unsigned char*)weighs)[weighPos] - 128);
					} else if (inputType == SIGN) {
						results[j] -= (((unsigned char*)weighs)[weighPos] - 128);
					}
				}
			}
		}
		inputOffset += getInput(i)->getWeighsSize();
//		printf("parciales ", 1);
//		for (unsigned i=0; i < output->getSize(); i++){
//			printf("%f ", results[i]);
//		}
//		printf("\n", 1);
	}


	printf("----------------\n", 1);
	for (unsigned i=0; i < output->getSize(); i++){
		printf("%f ", results[i]);
	}
	printf("\n----------------\n", 1);

	output->activation(results, functionType);
	mi_free(results);
}


