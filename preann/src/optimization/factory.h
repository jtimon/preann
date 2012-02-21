/*
 * factory.h
 *
 *  Created on: Mar 23, 2010
 *      Author: timon
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include "neural/connection.h"
#include "common/parametersMap.h"

class Factory
{
public:
    static const string SIZE;
    static const string WEIGHS_RANGE;
    static const string OUTPUT_SIZE;
protected:
    static BufferType weighForInput(BufferType inputType);
public:
    static void saveBuffer(Buffer* buffer, FILE* stream);
    static Buffer* newBuffer(FILE* stream, ImplementationType implementationType);

    static Buffer* newBuffer(Interface* interface, ImplementationType implementationType);
    static Buffer* newBuffer(Buffer* buffer, ImplementationType implementationType);
    static Buffer* newBuffer(unsigned size, BufferType bufferType, ImplementationType implementationType);
    static Connection* newConnection(Buffer* input, unsigned outputSize);
    static Connection* newThresholds(Buffer* output, ImplementationType implementationType);
    static Connection
            * newConnection(FILE* stream, unsigned outputSize, ImplementationType implementationType);

    static Buffer* newBuffer(ParametersMap* parametersMap);
    static Connection* newConnection(ParametersMap* parametersMap, Buffer* buffer);

};

#endif /* FACTORY_H_ */
