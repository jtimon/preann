
#include "cppLayer.h"

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


