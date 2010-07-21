#include "factory.h"

//TODO implementar diferentes veces para decidir en el makefile
#include "cppVector.h"
#include "cppLayer.h"
#include "xmmVector.h"
#include "xmmLayer.h"
#include "cudaVector.h"
#include "cudaLayer2.h"

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
