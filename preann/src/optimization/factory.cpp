#include "factory.h"
#include "configFactory.h"

Vector* Factory::newVector(FILE* stream, ImplementationType implementationType)
{
	Interface interface(stream);
	Vector* vector = Factory::newVector(&interface, implementationType);
	return vector;
}

Vector* Factory::newVector(Interface* interface, ImplementationType implementationType)
{
	Vector* toReturn = Factory::newVector(interface->getSize(), interface->getVectorType(), implementationType);
	toReturn->copyFromInterface(interface);
	return toReturn;
}

Vector* Factory::newVector(Vector* vector, ImplementationType implementationType)
{
    Interface* interface = vector->toInterface();
    Vector* toReturn = Factory::newVector(interface, implementationType);
    delete(interface);
    return toReturn;
}

Vector* Factory::newVector(unsigned size, VectorType vectorType, ImplementationType implementationType)
{
	switch(vectorType){
		case FLOAT:
			return func_newVector<FLOAT, float>(size, implementationType);
		case BYTE:
			return func_newVector<BYTE, unsigned char>(size, implementationType);
		case BIT:
			return func_newVector<BIT, unsigned>(size, implementationType);
		case SIGN:
			return func_newVector<SIGN, unsigned>(size, implementationType);
	}
}

Connection* Factory::newThresholds(Vector* output, ImplementationType implementationType)
{
	return func_newConnection<FLOAT, float>(output, 1, implementationType);
}

VectorType Factory::weighForInput(VectorType inputType)
{
	switch (inputType){
		case BYTE:
			{
			std::string error = "Connections are not implemented for an input Vector of the VectorType BYTE";
			throw error;
			}
		case FLOAT:
			return FLOAT;
		case BIT:
		case SIGN:
			return BYTE;
	}
}

Connection* Factory::newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	VectorType vectorType = Factory::weighForInput(input->getVectorType());

	switch(vectorType){
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
