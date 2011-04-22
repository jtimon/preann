
#ifndef CPPCONNECTION_H_
#define CPPCONNECTION_H_
#ifdef CPP_IMPL

#include "connection.h"
#include "cppVector.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class CppConnection: public virtual Connection, public CppVector<vectorTypeTempl, c_typeTempl> {
protected:
	virtual void mutateImpl(unsigned pos, float mutation)
	{
		switch (vectorTypeTempl){
		case BYTE:{
			c_typeTempl* weigh = &(((c_typeTempl*)data)[pos]);
			int result = (int)mutation + *weigh;
			if (result <= 0){
				*weigh = 0;
			}
			else if (result >= 255) {
				*weigh = 255;
			}
			else {
				*weigh = result;
			}
			}break;
		case FLOAT:
			((c_typeTempl*)data)[pos] += mutation;
			break;
		case BIT:
		case SIGN:
			{
			unsigned mask = 0x80000000>>(pos % BITS_PER_UNSIGNED) ;
			((unsigned*)data)[pos / BITS_PER_UNSIGNED] ^= mask;
			}
		}
	}

	virtual void crossoverImpl(Vector* other, Interface* bitVector)
	{
		switch (vectorTypeTempl){
			case BIT:
			case SIGN:
				{
				std::string error = "CppVector::crossoverImpl is not implemented for VectorType BIT nor SIGN.";
				throw error;
				}
			default:
			{
				c_typeTempl* otherWeighs = (c_typeTempl*)other->getDataPointer();
				c_typeTempl* thisWeighs = (c_typeTempl*)this->getDataPointer();
				c_typeTempl auxWeigh;

				for (unsigned i=0; i < tSize; i++){
					if (bitVector->getElement(i)){
						auxWeigh = thisWeighs[i];
						thisWeighs[i] = otherWeighs[i];
						otherWeighs[i] = auxWeigh;
					}
				}
			}
		}
	}

public:
	virtual ~CppConnection() {};

	CppConnection(Vector* input, unsigned outputSize): CppVector<vectorTypeTempl, c_typeTempl>(input->getSize() * outputSize)
	{
		tInput = input;
	}

	virtual void calculateAndAddTo(Vector* resultsVect)
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

};

#endif
#endif /* CPPCONNECTION_H_ */
