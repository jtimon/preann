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

template <VectorType vectorTypeTempl>
Vector* func_newVector(unsigned size, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppVector<vectorTypeTempl>(size);
		case SSE2:
			return new XmmVector<vectorTypeTempl>(size);
		case CUDA:
		case CUDA2:
		case CUDA_INV:
			return new CudaVector<vectorTypeTempl>(size);
	}
}

Vector* Factory::newVector(unsigned size, VectorType vectorType, ImplementationType implementationType)
{
	switch(vectorType){
		case FLOAT:
			return func_newVector<FLOAT>(size, implementationType);
		case BYTE:
			return func_newVector<BYTE>(size, implementationType);
		case BIT:
			return func_newVector<BIT>(size, implementationType);
		case SIGN:
			return func_newVector<SIGN>(size, implementationType);
	}
}

Connection* Factory::newConnection(FILE* stream, unsigned outputSize, ImplementationType implementationType)
{
	std::string error = "Factory::newConnection(FILE* ... deprecated";
	throw error;
}

template <VectorType vectorTypeTempl>
Connection* func_newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppConnection<vectorTypeTempl>(input, outputSize);
		case SSE2:
			return new XmmConnection<vectorTypeTempl>(input, outputSize);
		case CUDA:
			return new CudaConnection<vectorTypeTempl>(input, outputSize);
		case CUDA2:
			return new Cuda2Connection<vectorTypeTempl>(input, outputSize);
		case CUDA_INV:
			return new CudaInvertedConnection<vectorTypeTempl>(input, outputSize);
	}
}

Connection* Factory::newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	VectorType vectorType = Factory::weighForInput(input->getVectorType());

	switch(vectorType){
		case FLOAT:
			return func_newConnection<FLOAT>(input, outputSize, implementationType);
		case BYTE:
			return func_newConnection<BYTE>(input, outputSize, implementationType);
		case BIT:
			return func_newConnection<BIT>(input, outputSize, implementationType);
		case SIGN:
			return func_newConnection<SIGN>(input, outputSize, implementationType);
	}
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
