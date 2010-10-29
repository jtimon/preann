#include "factory.h"

//TODO implementar diferentes veces para decidir en el makefile
#include "cppVector.h"
#include "cppLayer.h"
#include "xmmVector.h"
#include "xmmLayer.h"
#include "cudaVector2.h"
#include "cudaLayer2.h"

void Factory::saveMatrix(Vector* matrix, FILE* stream, unsigned width, ImplementationType implementationType)
{
	Interface* interface = matrix->toInterface();

	if (implementationType == CUDA2){
		interface->transposeMatrix(width);
	}

	interface->save(stream);
	delete(interface);
}

Vector* Factory::newMatrix(FILE* stream, unsigned width, ImplementationType implementationType)
{
	Interface* interface = new Interface();
	interface->load(stream);

	if (implementationType == CUDA2){
		unsigned inputSize = interface->getSize() / width;
		interface->transposeMatrix(width);
	}

	Vector* vector = Factory::newVector(interface, implementationType);

	delete(interface);
	return vector;
}

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

Vector* Factory::newVector(unsigned size, VectorType vectorType, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppVector(size, vectorType);
		case SSE2:
			return new XmmVector(size, vectorType);
		case CUDA:
			return new CudaVector(size, vectorType);
		case CUDA2:
			return new CudaVector2(size, vectorType);
	}
}

Layer* Factory::newLayer(ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new CppLayer();
		case SSE2:
			return new XmmLayer();
		case CUDA:
			return new CudaLayer();
		case CUDA2:
			return new CudaLayer2();
	}
}

Layer* Factory::newLayer(unsigned size, VectorType outputType, ImplementationType implementationType, FunctionType functionType)
{
	Layer* toReturn;
	switch(implementationType){
		case C:
			toReturn = new CppLayer();
			break;
		case SSE2:
			toReturn = new XmmLayer();
			break;
		case CUDA:
			toReturn = new CudaLayer();
			break;
		case CUDA2:
			toReturn = new CudaLayer2();
			break;
	}
	toReturn->init(size, outputType, functionType);
	return toReturn;
}
