#include "factory.h"

//TODO Z implementar diferentes veces para decidir en el makefile
#include "template/cppConnection.h"
#include "template/xmmConnection.h"
#include "template/cuda2Connection.h"
#include "template/cudaInvertedConnection.h"

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

template <VectorType vectorTypeTempl, class c_typeTempl>
Vector* func_newVector(unsigned size, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppVector<vectorTypeTempl, c_typeTempl>(size);
		case SSE2:
			return new XmmVector<vectorTypeTempl, c_typeTempl>(size);
		case CUDA:
		case CUDA2:
		case CUDA_INV:
			return new CudaVector<vectorTypeTempl, c_typeTempl>(size);
	}
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

template <VectorType vectorTypeTempl, class c_typeTempl>
Connection* func_newConnection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case SSE2:
			return new XmmConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA:
			return new CudaConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA2:
			return new Cuda2Connection<vectorTypeTempl, c_typeTempl>(input, outputSize);
		case CUDA_INV:
			return new CudaInvertedConnection<vectorTypeTempl, c_typeTempl>(input, outputSize);
	}
}

Connection* Factory::newThresholds(Vector* output, ImplementationType implementationType)
{
	return func_newConnection<FLOAT, float>(output, 1, implementationType);
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
