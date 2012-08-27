/*
 * cppBuffer.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef CPPBUFFER_H_
#define CPPBUFFER_H_

#include "neural/buffer.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
class CppBuffer : virtual public Buffer
{
protected:
    unsigned getByteSize()
    {
        switch (bufferTypeTempl) {
            case BT_BIT:
            case BT_SIGN:
                return (((tSize - 1) / BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
            default:
                return tSize * sizeof(c_typeTempl);
        }
    }

    virtual void _copyFrom(Interface* interface)
    {
        memcpy(data, interface->getDataPointer(), interface->getByteSize());
    }

    virtual void _copyTo(Interface* interface)
    {
        memcpy(interface->getDataPointer(), data, this->getByteSize());
    }
public:
    CppBuffer()
    {
    }
    CppBuffer(unsigned size)
    {
        this->tSize = size;

        size_t byteSize = getByteSize();
        data = MemoryManagement::malloc(byteSize);

        reset();
    }

    ~CppBuffer()
    {
        if (data) {
            MemoryManagement::free(data);
            data = NULL;
        }
    }

    virtual ImplementationType getImplementationType()
    {
        return IT_C;
    }

    virtual BufferType getBufferType()
    {
        return bufferTypeTempl;
    }

    virtual void reset()
    {
        size_t byteSize = getByteSize();

        switch (bufferTypeTempl) {
            case BT_BYTE:
                SetValueToAnArray<c_typeTempl>(data, byteSize / sizeof(c_typeTempl), 128);
                break;
            default:
                SetValueToAnArray<c_typeTempl>(data, byteSize / sizeof(c_typeTempl), 0);
                break;
        }
    }
};

//TODO template specialization for BT_BYTE, BT_BIT and BT_SIGN
//template <class c_typeTempl>
//class CppBuffer<BT_BYTE, c_typeTempl>: virtual public Buffer{
//
//	void activation(Buffer* resultsVect, FunctionType functionType)
//	{
//		std::string error = "CppBuffer::activation is not implemented for BufferType BT_BYTE.";
//		throw error;
//	}
//};
//
////TODO juntar esto y lo siguiente (que es igual)
//template <class c_typeTempl>
//class CppBuffer<BT_BIT, c_typeTempl>: virtual public Buffer{
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
//class CppBuffer<BT_SIGN, c_typeTempl>: virtual public Buffer{
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
