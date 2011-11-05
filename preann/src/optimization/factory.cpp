#include "factory.h"
#include "configFactory.h"

#define FLOAT_STORAGE float
#define BYTE_STORAGE unsigned char
#define BIT_STORAGE unsigned
#define SIGN_STORAGE BIT_STORAGE

Buffer* Factory::newBuffer(FILE* stream, ImplementationType implementationType)
{
	Interface interface(stream);
	Buffer* buffer = Factory::newBuffer(&interface, implementationType);
	return buffer;
}

Buffer* Factory::newBuffer(Interface* interface, ImplementationType implementationType)
{
	Buffer* toReturn = Factory::newBuffer(interface->getSize(), interface->getBufferType(), implementationType);
	toReturn->copyFromInterface(interface);
	return toReturn;
}

Buffer* Factory::newBuffer(Buffer* buffer, ImplementationType implementationType)
{
    Interface* interface = buffer->toInterface();
    Buffer* toReturn = Factory::newBuffer(interface, implementationType);
    delete(interface);
    return toReturn;
}

Buffer* Factory::newBuffer(unsigned size, BufferType bufferType, ImplementationType implementationType)
{
	switch(bufferType){
		case BT_FLOAT:
			return func_newBuffer<BT_FLOAT, FLOAT_STORAGE>(size, implementationType);
		case BT_BYTE:
			return func_newBuffer<BT_BYTE, BYTE_STORAGE>(size, implementationType);
		case BT_BIT:
			return func_newBuffer<BT_BIT, BIT_STORAGE>(size, implementationType);
		case BT_SIGN:
			return func_newBuffer<BT_SIGN, SIGN_STORAGE>(size, implementationType);
	}
}

Connection* Factory::newThresholds(Buffer* output, ImplementationType implementationType)
{
	return func_newConnection<BT_FLOAT, FLOAT_STORAGE>(output, 1, implementationType);
}

BufferType Factory::weighForInput(BufferType inputType)
{
	switch (inputType){
		case BT_BYTE:
			{
			std::string error = "Connections are not implemented for an input Buffer of the BufferType BYTE";
			throw error;
			}
		case BT_FLOAT:
			return BT_FLOAT;
		case BT_BIT:
		case BT_SIGN:
			return BT_BYTE;
	}
}

Connection* Factory::newConnection(Buffer* input, unsigned outputSize, ImplementationType implementationType)
{
	BufferType bufferType = Factory::weighForInput(input->getBufferType());

	switch(bufferType){
		case BT_FLOAT:
			return func_newConnection<BT_FLOAT, FLOAT_STORAGE>(input, outputSize, implementationType);
		case BT_BYTE:
			return func_newConnection<BT_BYTE, BYTE_STORAGE>(input, outputSize, implementationType);
		case BT_BIT:
			return func_newConnection<BT_BIT, BIT_STORAGE>(input, outputSize, implementationType);
		case BT_SIGN:
			return func_newConnection<BT_SIGN, SIGN_STORAGE>(input, outputSize, implementationType);
	}
}
