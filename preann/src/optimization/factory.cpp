#include "factory.h"
#include "configFactory.h"

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
		case FLOAT:
			return func_newBuffer<FLOAT, float>(size, implementationType);
		case BYTE:
			return func_newBuffer<BYTE, unsigned char>(size, implementationType);
		case BIT:
			return func_newBuffer<BIT, unsigned>(size, implementationType);
		case SIGN:
			return func_newBuffer<SIGN, unsigned>(size, implementationType);
	}
}

Connection* Factory::newThresholds(Buffer* output, ImplementationType implementationType)
{
	return func_newConnection<FLOAT, float>(output, 1, implementationType);
}

BufferType Factory::weighForInput(BufferType inputType)
{
	switch (inputType){
		case BYTE:
			{
			std::string error = "Connections are not implemented for an input Buffer of the BufferType BYTE";
			throw error;
			}
		case FLOAT:
			return FLOAT;
		case BIT:
		case SIGN:
			return BYTE;
	}
}

Connection* Factory::newConnection(Buffer* input, unsigned outputSize, ImplementationType implementationType)
{
	BufferType bufferType = Factory::weighForInput(input->getBufferType());

	switch(bufferType){
		case FLOAT:
			return func_newConnection<FLOAT, float>(input, outputSize, implementationType);
		case BYTE:
			return func_newConnection<BYTE, unsigned char>(input, outputSize, implementationType);
		case BIT:
			return func_newConnection<BIT, unsigned>(input, outputSize, implementationType);
		case SIGN:
			return func_newConnection<SIGN, unsigned>(input, outputSize, implementationType);
	}
}
