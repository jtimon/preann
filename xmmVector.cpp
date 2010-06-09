#include "xmmVector.h"

XmmVector::XmmVector(unsigned size, VectorType vectorType, FunctionType functionType)
{
	this->size = size;
	this->vectorType = vectorType;
	switch (functionType){
		case FLOAT:
			this->functionType = functionType;
			break;
		case BIT:
			this->functionType = BINARY_STEP;
			break;
		case SIGN:
			this->functionType = BIPOLAR_STEP;
	}

	size_t byteSize = getByteSize();
	data = mi_malloc(byteSize);

	if (vectorType == FLOAT){

		unsigned floatSize = byteSize/sizeof(float);
		for (unsigned i=0; i< floatSize; i++){
			((float*)data)[i] = 0;
		}
	}
	else {

		for (unsigned i=0; i < byteSize; i++){
			((unsigned char*)data)[i] = 0;
		}
	}
}

XmmVector::~XmmVector()
{
	if (data) {
		mi_free(data);
		data = NULL;
	}
}

void XmmVector::copyFrom(Interface* interface)
{
	if (size < interface->getSize()){
		string error = "The Interface is greater than the Vector.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
		string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}

	if (vectorType == FLOAT){
		//memcpy(data, interface->getDataPointer(), interface->getByteSize());
		for (unsigned i=0; i < size; i++){
			if (interface->getElement(i) != 0){
				printf("no se debería tener un elemento no nulo en una interfaz\n");
			}
			((float*)data)[i] = interface->getElement(i);
		}
	} else {
		unsigned char* vectorData = (unsigned char*)data;
		unsigned blockOffset = 0;
		unsigned bytePos = 0;
		unsigned char vectorMask = 128;

		for (unsigned i=0; i < size; i++){

			if (interface->getElement(i) > 0){
				vectorData[blockOffset + bytePos] |= vectorMask;
			} else {
				vectorData[blockOffset + bytePos] &= ~vectorMask;
			}

			if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK-1)){
				bytePos = 0;
				if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK-1)){
					blockOffset += BYTES_PER_BLOCK;
					vectorMask = 128;
				} else {
					vectorMask >>= 1;
				}
			} else {
				++bytePos;
			}
		}
	}
}

void XmmVector::copyTo(Interface* interface)
{
	if (interface->getSize() < size){
		string error = "The Vector is greater than the Interface.";
		throw error;
	}
	if (vectorType != interface->getVectorType()){
		string error = "The Type of the Interface is different than the Vector Type.";
		throw error;
	}

	if (vectorType == FLOAT){
		//memcpy(interface->getDataPointer(), data, this->getByteSize());
		for (unsigned i=0; i < size; i++){
			interface->setElement(i, ((float*)data)[i]);
		}
	} else {
		unsigned char* vectorData = (unsigned char*)data;
		unsigned blockOffset = 0;
		unsigned bytePos = 0;
		unsigned char vectorMask = 128;

		for (unsigned i=0; i < size; i++){

			if (vectorData[blockOffset + bytePos] & vectorMask){
				interface->setElement(i, 1);
			} else {
				interface->setElement(i, 0);
			}

			if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK-1)){
				bytePos = 0;
				if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK-1)){
					blockOffset += BYTES_PER_BLOCK;
					vectorMask = 128;
				} else {
					vectorMask >>= 1;
				}
			} else {
				++bytePos;
			}
		}
	}
}

void XmmVector::activation(float* results)
{
	if (vectorType == FLOAT){
		for (unsigned i=0; i < size; i++){
			((float*)data)[i] = Function(results[i], functionType);
		}
	} else {
		unsigned char* vectorData = (unsigned char*)data;

		unsigned blockOffset = 0;
		unsigned bytePos = 0;
		unsigned char vectorMask = 128;

		for (unsigned i=0; i < size; i++){

			if (results[i] > 0){
				vectorData[blockOffset + bytePos] |= vectorMask;
			} else {
				vectorData[blockOffset + bytePos] &= ~vectorMask;
			}

			if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK-1)){
				bytePos = 0;
				if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK-1)){
					blockOffset += BYTES_PER_BLOCK;
					vectorMask = 128;
				} else {
					vectorMask >>= 1;
				}
			} else {
				++bytePos;
			}
		}
	}
	mi_free(results);
}

unsigned XmmVector::getByteSize()
{
	if (vectorType == FLOAT){
		return (((size-1)/FLOATS_PER_BLOCK)+1) * BYTES_PER_BLOCK;
	}
	else {
		return (((size-1)/BITS_PER_BLOCK)+1) * BYTES_PER_BLOCK;
	}
}
