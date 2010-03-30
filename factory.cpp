#include "factory.h"

//TODO implementar diferentes veces para decidir en el makefile
#include "cppLayer.h"
#include "xmmLayer.h"
#include "cudaLayer.h"

Vector* Factory::newVector(ImplementationType implementationType, unsigned size, VectorType vectorType = FLOAT)
{
	switch(implementationType){
		case C:
			return new Vector(size, vectorType);
		case SSE2:
			return new XmmVector(size, vectorType);
		case CUDA:
			return new CudaVector(size, vectorType);
	}
}

Layer* Factory::newLayer(ImplementationType implementationType, FunctionType functionType, VectorType inputType, VectorType outputType)
{
	switch(implementationType){
		case C:
			return new CppLayer(inputType, outputType, functionType);
		case SSE2:
			return new XmmLayer(inputType, outputType, functionType);
		case CUDA:
			return new CudaLayer(inputType, outputType, functionType);
	}
}
