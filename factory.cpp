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
	toReturn->copyFrom(interface);
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
			return new CudaVector(size, vectorType);
	}
}

Connection* Factory::newConnection(FILE* stream, unsigned outputSize, ImplementationType implementationType)
{
	std::string error = "Factory::newConnection(FILE* ... deprecated";
	throw error;
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
	switch(implementationType){
		case C:
			return new CppConnection(input, outputSize, Factory::weighForInput(input->getVectorType()));
		case SSE2:
			return new XmmConnection(input, outputSize, Factory::weighForInput(input->getVectorType()));
		case CUDA:
			return new CudaConnection(input, outputSize, Factory::weighForInput(input->getVectorType()));
		case CUDA2:
			return new Cuda2Connection(input, outputSize, Factory::weighForInput(input->getVectorType()));
		case CUDA_INV:
			return new CudaInvertedConnection(input, outputSize, Factory::weighForInput(input->getVectorType()));
	}
}
