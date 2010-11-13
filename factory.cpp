#include "factory.h"

//TODO implementar diferentes veces para decidir en el makefile
#include "cppVector.h"
#include "xmmVector.h"
#include "cudaVector2.h"

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
			return new CudaVector(size, vectorType);
		case CUDA2:
			return new CudaVector2(size, vectorType);
	}
}
