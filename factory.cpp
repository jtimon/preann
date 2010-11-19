#include "factory.h"

//TODO Z implementar diferentes veces para decidir en el makefile
#include "cppConnection.h"
#include "xmmConnection.h"
#include "cuda2Connection.h"
#include "cudaInvertedConnection.h"

void Factory::saveVector(Vector* vector, FILE* stream)
{
	Interface* interface = vector->toInterface();
	interface->save(stream);
	delete(interface);
}


Vector* Factory::newVector(FILE* stream, ImplementationType implementationType)
{
	Interface* interface = new Interface();
	interface->load(stream);

	Vector* vector = Factory::newVector(interface, implementationType);

	delete(interface);
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
	switch(implementationType){
		case C:
			return new CppVector(size, vectorType);
		case SSE2:
			return new XmmVector(size, vectorType);
		case CUDA:
		case CUDA2:
		case CUDA_INV:
			return new CudaVector(size, vectorType);
	}
}

Connection* Factory::newConnection(FILE* stream, unsigned outputSize, ImplementationType implementationType)
{
	std::string error = "Factory::newConnection(FILE* ... deprecated";
	throw error;
}

Connection* Factory::newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	VectorType vectorType = Factory::weighForInput(input->getVectorType());
	Connection* toReturn;
	switch(implementationType){
		case C:
			toReturn = new CppConnection(input, outputSize, vectorType);
			break;
		case SSE2:
			toReturn = new XmmConnection(input, outputSize, vectorType);
			break;
		case CUDA:
			toReturn = new CudaConnection(input, outputSize, vectorType);
			break;
		case CUDA2:
			toReturn = new Cuda2Connection(input, outputSize, vectorType);
			break;
		case CUDA_INV:
			toReturn = new CudaInvertedConnection(input, outputSize, vectorType);
			break;
	}
	return toReturn;
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
