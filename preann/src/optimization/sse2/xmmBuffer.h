/*
 * xmmBuffer.h
 *
 *  Created on: Nov 17, 2009
 *      Author: timon
 */

#ifndef XMMBUFFER_H_
#define XMMBUFFER_H_

#include "neural/buffer.h"
#include "sse2/sse2.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class XmmBuffer : virtual public Buffer
    {
    protected:
        static unsigned getByteSize(unsigned size)
        {
            unsigned numBlocks;
            switch (bufferTypeTempl) {
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

        virtual void _copyFrom(Interface* interface)
        {
            switch (bufferTypeTempl) {
                default:
                    memcpy(data, interface->getDataPointer(), interface->getByteSize());
                    break;
                case BT_BIT:
                case BT_SIGN:
                    unsigned char* bufferData = (unsigned char*) (data);
                    bitCopyFrom(bufferData, interface);
                    break;
            }
        }

        virtual void _copyTo(Interface* interface)
        {
            switch (bufferTypeTempl) {
                default:
                    memcpy(interface->getDataPointer(), data, interface->getByteSize());
                    break;
                case BT_BIT:
                case BT_SIGN:
                    unsigned char* bufferData = (unsigned char*) (data);
                    bitCopyTo(bufferData, interface);
                    break;
            }
        }

    public:
        XmmBuffer()
        {
        }
        XmmBuffer(unsigned size)
        {
            tSize = size;

            size_t byteSize = getByteSize(size);
            data = MemoryManagement::malloc(byteSize);

            reset();
        }

        ~XmmBuffer()
        {
            if (data) {
                MemoryManagement::free(data);
                data = NULL;
            }
        }

        virtual ImplementationType getImplementationType()
        {
            return IT_SSE2;
        }

        virtual BufferType getBufferType()
        {
            return bufferTypeTempl;
        }

        virtual void reset()
        {
            size_t byteSize = getByteSize(tSize);

            switch (bufferTypeTempl) {

                case BT_BYTE:
                    SetValueToAnArray<unsigned char>(data, byteSize, 128);
                    break;
                case BT_FLOAT:
                    SetValueToAnArray<float>(data, byteSize / sizeof(float), 0);
                    break;
                case BT_BIT:
                case BT_SIGN:
                    SetValueToAnArray<unsigned char>(data, byteSize, 0);
                    break;
            }
        }
    };

#endif /* XMMBUFFER_H_ */
