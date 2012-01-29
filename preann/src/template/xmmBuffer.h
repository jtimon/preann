/*
 * xmmBuffer.h
 *
 *  Created on: Nov 17, 2009
 *      Author: timon
 */

#ifndef XMMBUFFER_H_
#define XMMBUFFER_H_

#include "buffer.h"
#include "sse2_code.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class XmmBuffer : virtual public Buffer
    {
    protected:
        static unsigned getByteSize(unsigned size, BufferType bufferType)
        {
            unsigned numBlocks;
            switch (bufferType) {
                case BT_BYTE:
                    numBlocks = ((size - 1) / BYTES_PER_BLOCK) + 1;
                    break;
                case BT_FLOAT:
                    numBlocks = ((size - 1) / FLOATS_PER_BLOCK) + 1;
                    break;
                case BT_BIT:
                case BT_SIGN:
                    numBlocks = ((size - 1) / BITS_PER_BLOCK) + 1;
                    break;
            }
            return numBlocks * BYTES_PER_BLOCK;
        }

        void bitCopyFrom(unsigned char *bufferData, Interface *interface)
        {
            unsigned blockOffset = 0;
            unsigned bytePos = 0;
            unsigned char bufferMask = 128;
            for (unsigned i = 0; i < tSize; i++) {

                if (interface->getElement(i) > 0) {
                    bufferData[blockOffset + bytePos] |= bufferMask;
                } else {
                    bufferData[blockOffset + bytePos] &= ~bufferMask;
                }

                if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK - 1)) {
                    bytePos = 0;
                    if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK - 1)) {
                        blockOffset += BYTES_PER_BLOCK;
                        bufferMask = 128;
                    } else {
                        bufferMask >>= 1;
                    }
                } else {
                    ++bytePos;
                }
            }
        }

        void bitCopyTo(unsigned char *bufferData, Interface *interface)
        {
            unsigned blockOffset = 0;
            unsigned bytePos = 0;
            unsigned char bufferMask = 128;
            for (unsigned i = 0; i < tSize; i++) {

                if (bufferData[blockOffset + bytePos] & bufferMask) {
                    interface->setElement(i, 1);
                } else {
                    interface->setElement(i, 0);
                }

                if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK - 1)) {
                    bytePos = 0;
                    if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK - 1)) {
                        blockOffset += BYTES_PER_BLOCK;
                        bufferMask = 128;
                    } else {
                        bufferMask >>= 1;
                    }
                } else {
                    ++bytePos;
                }
            }
        }

        virtual void copyFromImpl(Interface* interface)
        {
            switch (bufferTypeTempl) {
                default:
                    memcpy(data, interface->getDataPointer(),
                            interface->getByteSize());
                    break;
                case BT_BIT:
                case BT_SIGN:
                    unsigned char* bufferData = (unsigned char*)(data);
                    bitCopyFrom(bufferData, interface);
            }
        }

        virtual void copyToImpl(Interface* interface)
        {
            switch (bufferTypeTempl) {
                default:
                    memcpy(interface->getDataPointer(), data,
                            interface->getByteSize());
                    break;
                case BT_BIT:
                case BT_SIGN:
                    unsigned char* bufferData = (unsigned char*)(data);
                    bitCopyTo(bufferData, interface);
                    break;
            }
        }

    public:
        virtual ImplementationType getImplementationType()
        {
            return IT_SSE2;
        }
        ;

        virtual BufferType getBufferType()
        {
            return bufferTypeTempl;
        }
        ;

        XmmBuffer()
        {
        }
        ;

        XmmBuffer(unsigned size)
        {
            this->tSize = size;

            size_t byteSize = getByteSize(size, bufferTypeTempl);
            data = MemoryManagement::malloc(byteSize);

            switch (bufferTypeTempl) {

                case BT_BYTE:
                    SetValueToAnArray<unsigned char> (data, byteSize, 128);
                    break;
                case BT_FLOAT:
                    SetValueToAnArray<float> (data, byteSize / sizeof(float), 0);
                    break;
                case BT_BIT:
                case BT_SIGN:
                    SetValueToAnArray<unsigned char> (data, byteSize, 0);
                    break;
            }
        }

        ~XmmBuffer()
        {
            if (data) {
                MemoryManagement::free(data);
                data = NULL;
            }
        }

        virtual Buffer* clone()
        {
            Buffer* clone = new XmmBuffer<bufferTypeTempl, c_typeTempl> (tSize);
            copyTo(clone);
            return clone;
        }

        virtual void activation(Buffer* resultsVect, FunctionType functionType)
        {
            float* results = (float*)resultsVect->getDataPointer();

            switch (bufferTypeTempl) {
                case BT_BYTE:
                    {
                        std::string error =
                                "XmmBuffer::activation is not implemented for BufferType BYTE.";
                        throw error;
                    }
                    break;
                case BT_FLOAT:
                    {
                        for (unsigned i = 0; i < tSize; i++) {
                            ((c_typeTempl*)data)[i] = Function<c_typeTempl> (
                                    results[i], functionType);
                        }
                    }
                    break;
                case BT_BIT:
                case BT_SIGN:
                    {
                        unsigned char* bufferData = (unsigned char*)data;

                        unsigned blockOffset = 0;
                        unsigned bytePos = 0;
                        unsigned char bufferMask = 128;

                        for (unsigned i = 0; i < tSize; i++) {

                            if (results[i] > 0) {
                                bufferData[blockOffset + bytePos] |= bufferMask;
                            } else {
                                bufferData[blockOffset + bytePos]
                                        &= ~bufferMask;
                            }

                            if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK - 1)) {
                                bytePos = 0;
                                if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK - 1)) {
                                    blockOffset += BYTES_PER_BLOCK;
                                    bufferMask = 128;
                                } else {
                                    bufferMask >>= 1;
                                }
                            } else {
                                ++bytePos;
                            }
                        }
                    }
            }
        }
    };

#endif /* XMMBUFFER_H_ */
