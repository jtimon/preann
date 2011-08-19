/*
 * cppBuffer.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef CPPBUFFER_H_
#define CPPBUFFER_H_

#include "buffer.h"

template <BufferType bufferTypeTempl, class c_typeTempl>
class CppBuffer: virtual public Buffer{
protected:
	unsigned getByteSize()
	{
		switch (bufferTypeTempl){
			case BIT:
			case SIGN:
				return (((tSize-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
			default:
				return tSize * sizeof(c_typeTempl);
		}
	}

	virtual void copyFromImpl(Interface* interface)
	{
		memcpy(data, interface->getDataPointer(), interface->getByteSize());
	}

	virtual void copyToImpl(Interface* interface)
	{
		memcpy(interface->getDataPointer(), data, this->getByteSize());
	}
public:

	virtual ImplementationType getImplementationType() {
		return C;
	};

	virtual BufferType getBufferType()
	{
		return bufferTypeTempl;
	};

	CppBuffer(){};

	CppBuffer(unsigned size)
	{
		this->tSize = size;

		size_t byteSize = getByteSize();
		data = mi_malloc(byteSize);

		switch (bufferTypeTempl){
			case BYTE:
				SetValueToAnArray<c_typeTempl>(data, byteSize/sizeof(c_typeTempl), 128);
				break;
			default:
				SetValueToAnArray<c_typeTempl>(data, byteSize/sizeof(c_typeTempl), 0);
		}
	}

	~CppBuffer()
	{
		if (data) {
			mi_free(data);
			data = NULL;
		}
	}

	virtual Buffer* clone()
	{
		Buffer* clone = new CppBuffer<bufferTypeTempl, c_typeTempl>(tSize);
		copyTo(clone);
		return clone;
	}

	virtual void activation(Buffer* resultsVect, FunctionType functionType)
	{
		float* results = (float*)resultsVect->getDataPointer();

		switch (bufferTypeTempl){
		case BYTE:
			{
				std::string error = "CppBuffer::activation is not implemented for BufferType BYTE.";
				throw error;
			}break;
		case FLOAT:
			{
				for (unsigned i=0; i < tSize; i++){
					((c_typeTempl*)data)[i] = Function<c_typeTempl>(results[i], functionType);
				}
			}
			break;
		case BIT:
		case SIGN:
			{
				unsigned* bufferData = (unsigned*)data;
				unsigned mask;
				for (unsigned i=0; i < tSize; i++){

					if (i % BITS_PER_UNSIGNED == 0){
						mask = 0x80000000;
					} else {
						mask >>= 1;
					}

					if (results[i] > 0){
						bufferData[i/BITS_PER_UNSIGNED] |= mask;
					} else {
						bufferData[i/BITS_PER_UNSIGNED] &= ~mask;
					}
				}
			}
		}
	}

};

#endif /* CPPBUFFER_H_ */
