#include "factory.h"

#include "xmmLayer.h"
#include "cudaLayer.h"
#include "cudaLayer2.h"

Vector* Factory::newVector(ImplementationType implementationType, unsigned size, VectorType vectorType = FLOAT)
{
	switch(implementationType){
		case C:
			printf("se construye vector C\n");
			return new Vector(size, vectorType);
		case SSE2:
			printf("se construye vector XMM\n");
			return new XmmVector(size, vectorType);
		case CUDA:
			printf("se construye vector C\n");
			return new Vector(size, vectorType);
		case CUDA2:
			printf("se construye vector CUDA\n");
			return new CudaVector(size, vectorType);
	}
}
/*
Layer* Factory::newLayer(ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			printf("se construye layer C\n");
			return new Layer();
		case SSE2:
			printf("se construye layer XMM\n");
			return new XmmLayer();
		case CUDA:
			printf("se construye layer CUDA\n");
			return new CudaLayer();
	}
}*/

Layer* Factory::newLayer(ImplementationType implementationType, FunctionType functionType, VectorType inputType, VectorType outputType)
{
	switch(implementationType){
		case C:
			printf("se construye layer C\n");
			return new Layer(inputType, outputType, functionType);
		case SSE2:
			printf("se construye layer XMM\n");
			return new XmmLayer(inputType, outputType, functionType);
		case CUDA:
			printf("se construye layer CUDA\n");
			return new CudaLayer(inputType, outputType, functionType);
		case CUDA2:
			printf("se construye layer CUDA\n");
			return new CudaLayer2(inputType, outputType, functionType);
	}
}
