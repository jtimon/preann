#include "factory.h"

//TODO implementar diferentes veces para decidir en el makefile
#include "cppLayer.h"
#include "xmmLayer.h"
#include "cudaLayer2.h"

Vector* Factory::newVector(unsigned size, VectorType vectorType, ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new Vector(size, vectorType);
		case SSE2:
			return new XmmVector(size, vectorType);
		case CUDA:
			return new CudaVector(size, vectorType);
		case CUDA2:
			return new CudaVector(size, vectorType);
	}
}

Layer* Factory::newLayer(unsigned size, VectorType outputType, ImplementationType implementationType, FunctionType functionType)
{
	switch(implementationType){
		case C:
			return new CppLayer(size, outputType, functionType);
		case SSE2:
			return new XmmLayer(size, outputType, functionType);
		case CUDA:
			return new CudaLayer(size, outputType, functionType);
		case CUDA2:
			return new CudaLayer2(size, outputType, functionType);
	}
}
