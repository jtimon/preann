#include "factory.h"

//TODO implementar diferentes veces para decidir en el makefile
#include "cppLayer.h"
#include "xmmLayer.h"
#include "cudaLayer2.h"

Vector* Factory::newVector(unsigned size, VectorType vectorType, ImplementationType implementationType, FunctionType functionType)
{
	switch(implementationType){
		case C:
			return new Vector(size, vectorType, functionType);
		case SSE2:
			return new XmmVector(size, vectorType, functionType);
		case CUDA:
			return new CudaVector(size, vectorType, functionType);
		case CUDA2:
			return new CudaVector(size, vectorType, functionType);
	}
}

Layer* Factory::newLayer(unsigned size, VectorType outputType, FunctionType functionType, ImplementationType implementationType)
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
