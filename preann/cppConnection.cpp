
#include "cppConnection.h"

CppConnection::CppConnection(Vector* input, unsigned outputSize, VectorType vectorType): CppVector(input->getSize() * outputSize, vectorType)
{
	tInput = input;
}

void CppConnection::calculateAndAddTo(Vector* resultsVect)
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
