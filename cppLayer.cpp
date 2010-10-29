
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


