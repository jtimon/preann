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
		data = MemoryManagement::malloc(byteSize);

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
			MemoryManagement::free(data);
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

//TODO template specialization for BYTE, BIT and SIGN
//template <class c_typeTempl>
//class CppBuffer<BYTE, c_typeTempl>: virtual public Buffer{
//
//	void activation(Buffer* resultsVect, FunctionType functionType)
//	{
//		std::string error = "CppBuffer::activation is not implemented for BufferType BYTE.";
//		throw error;
//	}
//};
//
////TODO juntar esto y lo siguiente (que es igual)
//template <class c_typeTempl>
//class CppBuffer<BIT, c_typeTempl>: virtual public Buffer{
//
//	void activation(Buffer* resultsVect, FunctionType functionType)
//	{
//		float* results = (float*)resultsVect->getDataPointer();
//		unsigned const bits_per_type = sizeof (c_typeTempl) * BITS_PER_BYTE;
//
//		c_typeTempl* bufferData = (c_typeTempl*)data;
//		c_typeTempl mask;
//		for (unsigned i=0; i < tSize; i++){
//			if (i % bits_per_type == 0){
//				//TODO quitar constante
//				mask = 0x80000000;
//			} else {
//				mask >>= 1;
//			}
//
//			if (results[i] > 0){
//				bufferData[i/bits_per_type] |= mask;
//			} else {
//				bufferData[i/bits_per_type] &= ~mask;
//			}
//		}
//	}
//
//	unsigned getByteSize()
//	{
//		return (((tSize-1)/(sizeof (c_typeTempl) * BITS_PER_BYTE))+1) * sizeof(c_typeTempl);
//	}
//};
//template <class c_typeTempl>
//class CppBuffer<SIGN, c_typeTempl>: virtual public Buffer{
//
//	void activation(Buffer* resultsVect, FunctionType functionType)
//	{
//		float* results = (float*)resultsVect->getDataPointer();
//		unsigned const bits_per_type = sizeof (c_typeTempl) * BITS_PER_BYTE;
//
//		c_typeTempl* bufferData = (c_typeTempl*)data;
//		c_typeTempl mask;
//		for (unsigned i=0; i < tSize; i++){
//			if (i % bits_per_type == 0){
//				//TODO quitar constante
//				mask = 0x80000000;
//			} else {
//				mask >>= 1;
//			}
//
//			if (results[i] > 0){
//				bufferData[i/bits_per_type] |= mask;
//			} else {
//				bufferData[i/bits_per_type] &= ~mask;
//			}
//		}
//	}
//
//	unsigned getByteSize()
//	{
//		return (((tSize-1)/(sizeof (c_typeTempl) * BITS_PER_BYTE))+1) * sizeof(c_typeTempl);
//	}
//};

#endif /* CPPBUFFER_H_ */
