
#ifndef CPPCONNECTION_H_
#define CPPCONNECTION_H_

#include "fullConnection.h"
#include "cppBuffer.h"

template <BufferType bufferTypeTempl, class c_typeTempl>
class CppConnection: public virtual FullConnection, public CppBuffer<bufferTypeTempl, c_typeTempl> {
protected:
	virtual void mutateImpl(unsigned pos, float mutation)
	{
		switch (bufferTypeTempl){
		case BT_BYTE:{
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
		case BT_FLOAT:
			((c_typeTempl*)data)[pos] += mutation;
			break;
		case BT_BIT:
		case BT_SIGN:
			{
			unsigned mask = 0x80000000>>(pos % BITS_PER_UNSIGNED) ;
			((unsigned*)data)[pos / BITS_PER_UNSIGNED] ^= mask;
			}
		}
	}

	virtual void crossoverImpl(Buffer* other, Interface* bitBuffer)
	{
		switch (bufferTypeTempl){
			case BT_BIT:
			case BT_SIGN:
				{
				std::string error = "CppBuffer::crossoverImpl is not implemented for BufferType BIT nor SIGN.";
				throw error;
				}
			default:
			{
				//TODO Z decidir cual mola mas
				c_typeTempl* otherWeighs = other->getDataPointer2<c_typeTempl>();
				c_typeTempl* thisWeighs = (c_typeTempl*)this->getDataPointer();
				c_typeTempl auxWeigh;

				for (unsigned i=0; i < tSize; i++){
					if (bitBuffer->getElement(i)){
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

	CppConnection(Buffer* input, unsigned outputSize): CppBuffer<bufferTypeTempl, c_typeTempl>(input->getSize() * outputSize)
	{
		tInput = input;
	}

	virtual void calculateAndAddTo(Buffer* resultsVect)
	{
		float* results = (float*)resultsVect->getDataPointer();
		unsigned inputSize = tInput->getSize();

		switch (tInput->getBufferType()){
		case BT_BYTE:
		{
			std::string error = "CppConnection::inputCalculation is not implemented for BufferType BYTE as input.";
			throw error;
		}
		case BT_FLOAT:
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
		case BT_BIT:
		case BT_SIGN:
		{
			unsigned char* inputWeighs = (unsigned char*)this->getDataPointer();
			unsigned* inputPtr = (unsigned*)tInput->getDataPointer();

			for (unsigned j=0; j < resultsVect->getSize(); j++){
				for (unsigned k=0; k < inputSize; k++){
					unsigned weighPos = (j * inputSize) + k;
					if ( inputPtr[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
						results[j] += inputWeighs[weighPos] - 128;
					} else if (tInput->getBufferType() == BT_SIGN) {
						results[j] -= inputWeighs[weighPos] - 128;
					}
				}
			}
		}
		break;
		}
	}

};

#endif /* CPPCONNECTION_H_ */
