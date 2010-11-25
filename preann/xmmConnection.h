#ifndef XMMCONNECTION_H_
#define XMMCONNECTION_H_

#include "connection.h"
#include "xmmVector.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class XmmConnection: virtual public Connection, public XmmVector<vectorTypeTempl, c_typeTempl> {
protected:
	//redefined from XmmVector
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);
public:
	XmmConnection(Vector* input, unsigned outputSize);
	virtual ~XmmConnection() {};

	virtual void calculateAndAddTo(Vector* results);

};

template <VectorType vectorTypeTempl, class c_typeTempl>
XmmConnection<vectorTypeTempl, c_typeTempl>::XmmConnection(Vector* input, unsigned outputSize)
{
	tInput = input;
	this->tSize = input->getSize() * outputSize;

	unsigned byteSize = XmmVector<vectorTypeTempl, c_typeTempl>::getByteSize(input->getSize(), vectorTypeTempl);
	byteSize *= outputSize;
	data = mi_malloc(byteSize);

	switch (vectorTypeTempl){

	case BYTE:
		SetValueToAnArray<unsigned char>(data, byteSize, 128);
		break;
	case FLOAT:
		SetValueToAnArray<float>(data, byteSize/sizeof(float), 0);
		break;
	case BIT:
	case SIGN:
		SetValueToAnArray<unsigned char>(data, byteSize, 0);
		break;
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmConnection<vectorTypeTempl, c_typeTempl>::calculateAndAddTo(Vector* resultsVect)
{
	void* inputWeighs = this->getDataPointer();
	float* results = (float*)resultsVect->getDataPointer();
	void* inputPtr = tInput->getDataPointer();

	unsigned numLoops;
	unsigned offsetPerInput = XmmVector<vectorTypeTempl, c_typeTempl>::getByteSize(tInput->getSize(), vectorTypeTempl);
	unsigned weighPos = 0;


	switch (tInput->getVectorType()){
		case FLOAT:
			offsetPerInput = offsetPerInput / sizeof(float);
			numLoops = ((tInput->getSize()-1)/FLOATS_PER_BLOCK)+1;
			for (unsigned j=0; j < resultsVect->getSize(); j++){
				float auxResult;
				XMMreal(inputPtr, numLoops,
						(((float*)inputWeighs) + weighPos), auxResult);
				results[j] += auxResult;
				weighPos += offsetPerInput;
			}
			break;
		case BIT:
			numLoops = ((tInput->getSize()-1)/BYTES_PER_BLOCK)+1;
			for (unsigned j=0; j < resultsVect->getSize(); j++){
				results[j] += XMMbinario(inputPtr, numLoops,
						(((unsigned char*)inputWeighs) + weighPos));
				weighPos += offsetPerInput;
			}
			break;
		case SIGN:
			numLoops = ((tInput->getSize()-1)/BYTES_PER_BLOCK)+1;
			for (unsigned j=0; j < resultsVect->getSize(); j++){
				results[j] += XMMbipolar(inputPtr, numLoops,
									(((unsigned char*)inputWeighs) + weighPos));
				weighPos += offsetPerInput;
			}
			break;
		case BYTE:
			std::string error = "CppVector::inputCalculation is not implemented for VectorType BYTE as input.";
			throw error;
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmConnection<vectorTypeTempl, c_typeTempl>::copyFromImpl(Interface* interface)
{
	unsigned offsetPerInput = XmmVector<vectorTypeTempl, c_typeTempl>::getByteSize(tInput->getSize(), vectorTypeTempl);
	unsigned offset = 0;
	unsigned inputSize = tInput->getSize();
	unsigned outputSize = tSize / inputSize;
	unsigned elem = 0;

	switch (vectorTypeTempl){
		case BYTE:
			for (unsigned j=0; j < outputSize; j++){
				for (unsigned i=0; i < inputSize; i++){
					((unsigned char*)(data) + offset)[i] = interface->getElement(elem++);
				}
				offset += offsetPerInput;
			}
			break;
		case FLOAT:
			offsetPerInput = offsetPerInput / sizeof(float);
			for (unsigned j=0; j < outputSize; j++){
				for (unsigned i=0; i < inputSize; i++){
					((float*)(data) + offset)[i] = interface->getElement(elem++);
				}
				offset += offsetPerInput;
			}
			break;
		case BIT:
		case SIGN:
		{
			std::string error = "XmmConnection::copyFromImpl is not implemented for VectorType BIT nor SIGN.";
			throw error;
		}
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmConnection<vectorTypeTempl, c_typeTempl>::copyToImpl(Interface* interface)
{
	unsigned offsetPerInput = XmmVector<vectorTypeTempl, c_typeTempl>::getByteSize(tInput->getSize(), vectorTypeTempl);
	unsigned offset = 0;
	unsigned inputSize = tInput->getSize();
	unsigned outputSize = tSize / inputSize;
	unsigned elem = 0;

	switch (vectorTypeTempl){
		case BYTE:
			for (unsigned j=0; j < outputSize; j++){
				for (unsigned i=0; i < inputSize; i++){
					interface->setElement(elem++, ((unsigned char*)(data) + offset)[i]);
				}
				offset += offsetPerInput;
			}
			break;
		case FLOAT:
			offsetPerInput = offsetPerInput / sizeof(float);
			for (unsigned j=0; j < outputSize; j++){
				for (unsigned i=0; i < inputSize; i++){
					interface->setElement(elem++, ((float*)(data) + offset)[i]);
				}
				offset += offsetPerInput;
			}
			break;
		case BIT:
		case SIGN:
			{
				std::string error = "XmmConnection::copyToImpl is not implemented for VectorType BIT nor SIGN.";
				throw error;
			}
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmConnection<vectorTypeTempl, c_typeTempl>::mutateImpl(unsigned pos, float mutation)
{
	unsigned offsetPerInput = XmmVector<vectorTypeTempl, c_typeTempl>::getByteSize(tInput->getSize(), vectorTypeTempl);
	unsigned outputPos = pos / tInput->getSize();
    unsigned inputPos = pos % tInput->getSize();
    unsigned elem = (outputPos * offsetPerInput) + inputPos;

    switch (vectorTypeTempl){
        case BYTE:
            {
                unsigned char *weigh = &(((unsigned char*)(data))[elem]);
                int result = (int)(mutation) + *weigh;
                if(result <= 0){
                    *weigh = 0;
            }else
                if(result >= 255){
                    *weigh = 255;
                }else{
                    *weigh = result;
                }

        }
        break;
    case FLOAT:
    	offsetPerInput = offsetPerInput / sizeof(float);
    	elem = (outputPos * offsetPerInput) + inputPos;
        ((float*)(data))[elem] += mutation;
			break;
	case BIT:
	case SIGN:
		std::string error = "XmmConnection::mutate is not implemented for VectorType BIT nor SIGN.";
		throw error;
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void XmmConnection<vectorTypeTempl, c_typeTempl>::crossoverImpl(Vector* other, Interface* bitVector)
{
	void* otherWeighs = other->getDataPointer();

	unsigned offsetPerInput = XmmVector<vectorTypeTempl, c_typeTempl>::getByteSize(tInput->getSize(), vectorTypeTempl);
	unsigned offset = 0;
	unsigned inputSize = tInput->getSize();
	unsigned outputSize = tSize / inputSize;
	unsigned elem = 0;

	switch (vectorTypeTempl){
		case BYTE:
			unsigned char auxChar;

			for (unsigned j=0; j < outputSize; j++){
				for (unsigned i=0; i < inputSize; i++){

					if (bitVector->getElement(elem++)){
						auxChar = ((unsigned char*)(data) + offset)[i];
						((unsigned char*)(data) + offset)[i] = ((unsigned char*)(otherWeighs) + offset)[i];
						((unsigned char*)(otherWeighs) + offset)[i] = auxChar;
					}
				}
				offset += offsetPerInput;
			}
			break;
		case FLOAT:
			float auxFloat;
			offsetPerInput = offsetPerInput / sizeof(float);

			for (unsigned j=0; j < outputSize; j++){
				for (unsigned i=0; i < inputSize; i++){

					if (bitVector->getElement(elem++)){
						auxFloat = ((float*)(data) + offset)[i];
						((float*)(data) + offset)[i] = ((float*)(otherWeighs) + offset)[i];
						((float*)(otherWeighs) + offset)[i] = auxFloat;
					}
				}
				offset += offsetPerInput;
			}
			break;
		case BIT:
		case SIGN:
			std::string error = "XmmConnection::crossoverImpl is not implemented for VectorType BIT nor SIGN.";
			throw error;
	}
}


#endif /* XMMCONNECTION_H_ */
