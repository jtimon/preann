
#include "cppLayer.h"

CppLayer::CppLayer(VectorType outputType, FunctionType functionType): Layer(outputType, functionType)
{
}

CppLayer::CppLayer(unsigned size, VectorType outputType, FunctionType functionType): Layer(outputType, functionType)
{
	output = new Vector(size, outputType);
	thresholds = (float*)mi_malloc(sizeof(float) * size);
}

CppLayer::~CppLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			mi_free(weighs[i]);
		}
		mi_free(inputs);
		mi_free(weighs);
		inputs = NULL;
		weighs = NULL;
	}
	if (thresholds) {
		mi_free(thresholds);
		thresholds = NULL;
	}
	if (output) {
		delete (output);
		output = NULL;
	}
}

void CppLayer::saveWeighs(FILE* stream)
{
	fwrite(thresholds, output->getSize() * sizeof(float), 1, stream);
	for (unsigned i=0; i < numberInputs; i++){
		unsigned size;
		if (inputs[i]->getVectorType() == FLOAT){
			size = output->getSize() * inputs[i]->getSize() * sizeof(float);
		} else {
			size = output->getSize() * inputs[i]->getSize() * sizeof(unsigned char);
		}
		fwrite(weighs[i], size, 1, stream);
	}
}

void CppLayer::loadWeighs(FILE* stream)
{
	fread(thresholds, output->getSize() * sizeof(float), 1, stream);
	for (unsigned i=0; i < numberInputs; i++){
		unsigned size;
		if (inputs[i]->getVectorType() == FLOAT){
			size = output->getSize() * inputs[i]->getSize() * sizeof(float);
		} else {
			size = output->getSize() * inputs[i]->getSize() * sizeof(unsigned char);
		}
		fread(weighs[i], size, 1, stream);
	}
}

float* CppLayer::negativeThresholds()
{
	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	for (unsigned j=0; j < output->getSize(); j++) {
		results[j] = -thresholds[j];
	}
	return results;
}

void CppLayer::inputCalculation(Vector* input, void* inputWeighs, float* results)
{
	void* inputPtr = input->getDataPointer();

	for (unsigned j=0; j < output->getSize(); j++){

		for (unsigned k=0; k < input->getSize(); k++){
			unsigned weighPos = (j * input->getSize()) + k;
			if (input->getVectorType() == FLOAT) {
				//printf("i % d input %f weigh %f \n", k, ((float*)input)[k], ((float*)weighs)[weighPos]);
				results[j] += ((float*)inputPtr)[k] * ((float*)inputWeighs)[weighPos];
			} else {
				if ( ((unsigned*)inputPtr)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
					results[j] += (((unsigned char*)inputWeighs)[weighPos] - 128);
				} else if (input->getVectorType() == SIGN) {
					results[j] -= (((unsigned char*)inputWeighs)[weighPos] - 128);
				}
			}
		}
	}
}

void* CppLayer::newWeighs(unsigned inputSize, VectorType inputType)
{
	unsigned size;
	if (inputType == FLOAT) {
		size = output->getSize() * inputSize * sizeof(float);
	} else {
		size = output->getSize() * inputSize * sizeof(unsigned char);
	}
	return mi_malloc(size);
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
	unsigned charRange;
	if (range >= 128){
		charRange = 127;
	} else {
		charRange = (unsigned)range;
	}
	for (unsigned i=0; i < output->getSize(); i++){

		//thresholds[i] = 0;
		thresholds[i] = randomFloat(range);
		for (unsigned j=0; j < numberInputs; j++){

			unsigned inputSize = inputs[j]->getSize();
			for (unsigned k=0; k < inputSize; k++){

				unsigned weighPos = (i * inputSize) + k;
				if (inputs[j]->getVectorType() == FLOAT) {
					((float**)weighs)[j][weighPos] = randomFloat(range);
				} else {
					((unsigned char**)weighs)[j][weighPos] = 128 + (unsigned char)randomInt(charRange);
				}
			}
		}
	}
}

/*
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

	for (unsigned i=0; i < numberInputs; i++){

		void* input = inputs[i]->getDataPointer();

		for (unsigned j=0; j < output->getSize(); j++){

			for (unsigned k=0; k < inputs[i]->getSize(); k++){
				unsigned weighPos = (j * inputs[i]->getSize()) + k;
				if (inputType == FLOAT) {
					//printf("i % d input %f weigh %f \n", k, ((float*)input)[k], ((float*)weighs)[weighPos]);
					results[j] += ((float*)input)[k] * ((float**)weighs)[i][weighPos];
				} else {
					if ( ((unsigned*)input)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
						results[j] += (((unsigned char**)weighs)[i][weighPos] - 128);
					} else if (inputType == SIGN) {
						results[j] -= (((unsigned char**)weighs)[i][weighPos] - 128);
					}
				}
			}
		}
	}

//	printf("----------------\n", 1);
//	for (unsigned i=0; i < output->getSize(); i++){
//		printf("%f ", results[i]);
//	}
//	printf("\n----------------\n", 1);

	output->activation(results, functionType);
	mi_free(results);
}*/

Layer* CppLayer::newCopy()
{
	std::string error = "newCopy is not implemented for CppLayer.";
	throw error;
}
