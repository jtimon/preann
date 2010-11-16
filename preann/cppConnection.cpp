
#include "cppConnection.h"

CppConnection::CppConnection(Vector* input, unsigned outputSize, VectorType vectorType): CppVector(input->getSize() * outputSize, vectorType)
{
	tInput = input;
}

//TODO D igual que en la version Vector
void CppConnection::crossover(Connection* other, Interface* bitVector)
{
	if (size != other->getSize()){
		std::string error = "The Connections must have the same size to crossover them.";
		throw error;
	}
	if (vectorType != other->getVectorType()){
		std::string error = "The Connections must have the same type to crossover them.";
		throw error;
	}

	void* otherWeighs = other->getDataPointer();
	void* thisWeighs = this->getDataPointer();

	switch (vectorType){
	case BYTE:{
		unsigned char auxWeigh;

		for (unsigned i=0; i < size; i++){

			if (bitVector->getElement(i)){
				auxWeigh = ((unsigned char*)thisWeighs)[i];
				((unsigned char*)thisWeighs)[i] = ((unsigned char*)otherWeighs)[i];
				((unsigned char*)otherWeighs)[i] = auxWeigh;
			}
		}
		}break;
	case FLOAT:
		float auxWeigh;

		for (unsigned i=0; i < size; i++){

			if (bitVector->getElement(i)){
				auxWeigh = ((float*)thisWeighs)[i];
				((float*)thisWeighs)[i] = ((float*)otherWeighs)[i];
				((float*)otherWeighs)[i] = auxWeigh;
			}
		}
		break;
	case BIT:
	case SIGN:
		{
		std::string error = "CppConnection::weighCrossover is not implemented for VectorType BIT nor SIGN.";
		throw error;
		}
	}
}

void CppConnection::addToResults(Vector* resultsVect)
{
	float* results = (float*)resultsVect->getDataPointer();
	unsigned inputSize = tInput->getSize();

	switch (tInput->getVectorType()){
	case BYTE:
	{
		std::string error = "CppConnection::inputCalculation is not implemented for VectorType BYTE as input.";
		throw error;
	}
	case FLOAT:
	{
		float* inputWeighs = (float*)this->getDataPointer();
		float* inputPtr = (float*)tInput->getDataPointer();
		for (unsigned j=0; j < resultsVect->getSize(); j++){
			for (unsigned k=0; k < inputSize; k++){
				results[j] += inputPtr[k] * inputWeighs[(j * inputSize) + k];
			}
		}
	}
	break;
	case BIT:
	case SIGN:
	{
		unsigned char* inputWeighs = (unsigned char*)this->getDataPointer();
		unsigned* inputPtr = (unsigned*)tInput->getDataPointer();

		for (unsigned j=0; j < resultsVect->getSize(); j++){
			for (unsigned k=0; k < inputSize; k++){
				unsigned weighPos = (j * inputSize) + k;
				if ( inputPtr[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
					results[j] += inputWeighs[weighPos] - 128;
				} else if (tInput->getVectorType() == SIGN) {
					results[j] -= inputWeighs[weighPos] - 128;
				}
			}
		}
	}
	break;
	}
}
