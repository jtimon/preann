#include "factory.h"
#include "configFactory.h"

//TODO BP usar esto en tos laos y cambiar de sitio
#define FLOAT_STORAGE float
#define BYTE_STORAGE unsigned char
#define BIT_STORAGE unsigned
#define SIGN_STORAGE BIT_STORAGE

const string Factory::SIZE = "__size";
const string Factory::WEIGHS_RANGE = "__initialWeighsRange";
const string Factory::OUTPUT_SIZE = "__outputSize";

Buffer* Factory::newBuffer(FILE* stream, ImplementationType implementationType)
{
    Interface interface(stream);
    Buffer* buffer = Factory::newBuffer(&interface, implementationType);
    return buffer;
}

Buffer* Factory::newBuffer(Interface* interface,
        ImplementationType implementationType)
{
    Buffer* toReturn = Factory::newBuffer(interface->getSize(),
            interface->getBufferType(), implementationType);
    toReturn->copyFromInterface(interface);
    return toReturn;
}

Buffer* Factory::newBuffer(Buffer* buffer,
        ImplementationType implementationType)
{
    Interface* interface = buffer->toInterface();
    Buffer* toReturn = Factory::newBuffer(interface, implementationType);
    delete (interface);
    return toReturn;
}

Buffer* Factory::newBuffer(unsigned size, BufferType bufferType,
        ImplementationType implementationType)
{
    switch (bufferType) {
        case BT_FLOAT:
            return func_newBuffer<BT_FLOAT, FLOAT_STORAGE> (size,
                    implementationType);
        case BT_BYTE:
            return func_newBuffer<BT_BYTE, BYTE_STORAGE> (size,
                    implementationType);
        case BT_BIT:
            return func_newBuffer<BT_BIT, BIT_STORAGE> (size,
                    implementationType);
        case BT_SIGN:
            return func_newBuffer<BT_SIGN, SIGN_STORAGE> (size,
                    implementationType);
    }
}

Connection* Factory::newThresholds(Buffer* output,
        ImplementationType implementationType)
{
    return func_newConnection<BT_FLOAT, FLOAT_STORAGE> (output, 1,
            implementationType);
}

BufferType Factory::weighForInput(BufferType inputType)
{
    switch (inputType) {
        case BT_BYTE:
            {
                std::string error =
                        "Connections are not implemented for an input Buffer of the BufferType BYTE";
                throw error;
            }
        case BT_FLOAT:
            return BT_FLOAT;
        case BT_BIT:
        case BT_SIGN:
            return BT_BYTE;
    }
}

Connection* Factory::newConnection(Buffer* input, unsigned outputSize)
{
    BufferType bufferType = Factory::weighForInput(input->getBufferType());
    ImplementationType implementationType = input->getImplementationType();

    switch (bufferType) {
        case BT_FLOAT:
            return func_newConnection<BT_FLOAT, FLOAT_STORAGE> (input,
                    outputSize, implementationType);
        case BT_BYTE:
            return func_newConnection<BT_BYTE, BYTE_STORAGE> (input,
                    outputSize, implementationType);
        case BT_BIT:
            return func_newConnection<BT_BIT, BIT_STORAGE> (input, outputSize,
                    implementationType);
        case BT_SIGN:
            return func_newConnection<BT_SIGN, SIGN_STORAGE> (input,
                    outputSize, implementationType);
    }
}

Buffer* Factory::newBuffer(ParametersMap* parametersMap)
{
    BufferType bufferType = (BufferType)(parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER)));
    ImplementationType implementationType = (ImplementationType)(parametersMap->getNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION)));

    unsigned size = (unsigned )(parametersMap->getNumber(Factory::SIZE));
    float initialWeighsRange = parametersMap->getNumber(Factory::WEIGHS_RANGE);

    Buffer* buffer = Factory::newBuffer(size, bufferType, implementationType);
    buffer->random(initialWeighsRange);

    return buffer;
}

Connection* Factory::newConnection(ParametersMap* parametersMap, Buffer* buffer)
{
    float initialWeighsRange = parametersMap->getNumber(Factory::WEIGHS_RANGE);
    unsigned outputSize = parametersMap->getNumber(Factory::OUTPUT_SIZE);

    Connection* connection = Factory::newConnection(buffer, outputSize);
    connection->random(initialWeighsRange);

    return connection;
}
