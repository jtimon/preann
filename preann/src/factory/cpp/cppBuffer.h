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

#endif /* CPPBUFFER_H_ */
