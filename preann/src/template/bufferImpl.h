/*
 * bufferImpl.h
 *
 *  Created on: Nov 23, 2010
 *      Author: timon
 */

#ifndef BUFFERIMPL_H_
#define BUFFERIMPL_H_

#include "buffer.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class BufferImpl : virtual public Buffer
    {
    protected:
        BufferImpl()
        {
        }
        ;
    public:
        virtual ~BufferImpl()
        {
        }
        ;

        BufferType getBufferType()
        {
            return bufferTypeTempl;
        }
        c_typeTempl* getDataPointer2()
        {
            return (c_typeTempl*)data;
        }
        ;
    };

#endif /* BUFFERIMPL_H_ */
