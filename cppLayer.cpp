
#include "cppLayer.h"

CppLayer::CppLayer()
{
}

CppLayer::~CppLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			delete(connections[i]);
		}

		mi_free(inputs);
		mi_free(connections);
		inputs = NULL;
		connections = NULL;
	}
	if (thresholds) {
		delete(thresholds);
		thresholds = NULL;
	}
	if (output) {
		delete (output);
		output = NULL;
	}
}

float* CppLayer::negativeThresholds()
{
	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	float* thresholdsDataPtr = (float*) thresholds->getDataPointer();
	for (unsigned j=0; j < output->getSize(); j++) {
		results[j] = - thresholdsDataPtr[j];
	}
	return results;
}

void CppLayer::inputCalculation(Vector* input, void* inputWeighs, float* results)
{
	void* inputPtr = input->getDataPointer();
	unsigned inputSize = input->getSize();

	for (unsigned j=0; j < output->getSize(); j++){

		for (unsigned k=0; k < inputSize; k++){

			unsigned weighPos = (j * inputSize) + k;
			if (input->getVectorType() == FLOAT) {
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

void CppLayer::copyWeighs(Layer* sourceLayer)
{
	memcpy(thresholds, sourceLayer->getThresholdsPtr(), output->getSize() * sizeof(float));

	for (unsigned i=0; i < numberInputs; i++){
		//TODO implementar metodo
	}
}

void CppLayer::crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector)
{
	unsigned weighsSize = bitVector->getSize();

	if (inputs[inputLayer]->getVectorType() == FLOAT){

		float* otherWeighs = (float*)(other->getConnection(inputLayer));
		float* thisWeighs = (float*)(this->getConnection(inputLayer));
		float auxWeigh;

		for (unsigned i=0; i < weighsSize; i++){

			if (bitVector->getElement(i)){
				auxWeigh = thisWeighs[i];
				thisWeighs[i] = otherWeighs[i];
				otherWeighs[i] = auxWeigh;
			}
		}
	} else {
		unsigned char* otherWeighs = (unsigned char*)(other->getConnection(inputLayer));
		unsigned char* thisWeighs = (unsigned char*)(this->getConnection(inputLayer));
		unsigned char auxWeigh;

		for (unsigned i=0; i < weighsSize; i++){

			if (bitVector->getElement(i)){
				auxWeigh = thisWeighs[i];
				thisWeighs[i] = otherWeighs[i];
				otherWeighs[i] = auxWeigh;
			}
		}
	}
}


