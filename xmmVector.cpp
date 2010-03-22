

/*
 * xmmVector.cpp
 *
 *  Created on: Nov 17, 2009
 *      Author: timon
 */

#include "xmmVector.h"

XmmVector::XmmVector(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

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
			((unsigned char*)data)[i] = 255;
		}
	}
}

XmmVector::~XmmVector()
{
}

unsigned XmmVector::getByteSize()
{
	if (vectorType == FLOAT){

		return (((size-1)/FLOATS_PER_BLOCK)+1) * FLOATS_PER_BLOCK * sizeof(float);
	}
	else {
		return (((size-1)/BITS_PER_BLOCK)+1) * BYTES_PER_BLOCK;
	}
}

unsigned XmmVector::posToBytePos(unsigned  pos)
{
	return pos%BYTES_PER_BLOCK + ((pos/BITS_PER_BLOCK)*BYTES_PER_BLOCK);
}

unsigned XmmVector::posToBitPos(unsigned  pos)
{
	return (pos/BYTES_PER_BLOCK)%BITS_PER_BYTE;
}

void XmmVector::setElement(unsigned  pos, float value)
{
	if (pos >= size){
		cout<<"Error: trying to access a position greater than the vector size."<<endl;
	}
	else {
		if (vectorType == FLOAT){

			((float*)data)[pos] = value;
		} else {
			unsigned bytePos = posToBytePos(pos);
			unsigned bitPos = posToBitPos(pos);

			unsigned char mask = (unsigned char)(0x80>>bitPos);
			if (value == 1){
				((unsigned char*)data)[bytePos] = ((unsigned char*)data)[bytePos] | mask;
			} else if (value == 0 || value == -1) {
				((unsigned char*)data)[bytePos] = ((unsigned char*)data)[bytePos] & ~mask;
			}
			else {
				cout<<"Error: A float value cannot be assigned to a bit nor a sign element."<<endl;
			}
		}
	}
}

float XmmVector::getElement(unsigned  pos)
{
	if (pos >= size){
		cout<<"Error: trying to access a position greater than the vector size."<<endl;
		return 0;
	} else {
		if (vectorType == FLOAT){

			return ((float*)data)[pos];
		}
		else {
			unsigned bytePos = posToBytePos(pos);
			unsigned bitPos = posToBitPos(pos);

			unsigned char mask = (unsigned char)(0x80>>bitPos);
			if (((unsigned char*)data)[bytePos] & mask){
				return 1;
			}
			else {
				if (vectorType == BIT) {
					return 0;
				}
				else{
					return -1;
				}
			}
		}
	}
}

unsigned XmmVector::getNumLoops()
{
	unsigned toReturn;

	if (vectorType == FLOAT){
		toReturn = ((size-1)/FLOATS_PER_BLOCK)+1;
	} else {
		toReturn = ((size-1)/BYTES_PER_BLOCK)+1;
	}
	return toReturn;
}

