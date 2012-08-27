/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include "neural/connection.h"

class Factory
{
protected:
    static BufferType weighForInput(BufferType inputType);
public:
    static void saveBuffer(Buffer* buffer, FILE* stream);
    static Buffer* newBuffer(FILE* stream, ImplementationType implementationType);

    static Buffer* newBuffer(Interface* interface, ImplementationType implementationType);
    static Buffer* newBuffer(Buffer* buffer, ImplementationType implementationType);
    static Buffer* newBuffer(unsigned size, BufferType bufferType, ImplementationType implementationType);
    static Connection* newConnection(Buffer* input, unsigned outputSize);
    static Connection* newConnection(FILE* stream, unsigned outputSize,
                                     ImplementationType implementationType);

};

#endif /* FACTORY_H_ */
