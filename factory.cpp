#include "factory.h"

#include "xmmLayer.h"
#include "cudaLayer.h"

Factory::Factory() {
	// TODO Auto-generated constructor stub

}

Factory::~Factory() {
	// TODO Auto-generated destructor stub
}

Vector* Factory::newVector(ImplementationType implementationType, unsigned size, VectorType vectorType)
{
	switch(implementationType){
		case C:
		case CUDA:
			printf("se construye vector C\n");
			return new Vector(size, vectorType);
		case SSE2:
			printf("se construye vector XMM\n");
			return new XmmVector(size, vectorType);
	}
}

Layer* Factory::newLayer(ImplementationType implementationType)
{
	switch(implementationType){
		case C:
			return new Layer();
		case SSE2:
			return new XmmLayer();
		case CUDA:
			return new CudaLayer();
	}
}

Layer* Factory::newLayer(ImplementationType implementationType, VectorType inputType, VectorType outputType, FunctionType functionType)
{
	switch(implementationType){
		case C:
			return new Layer(inputType, outputType, functionType);
		case SSE2:
			return new XmmLayer(inputType, outputType, functionType);
		case CUDA:
			return new CudaLayer(inputType, outputType, functionType);
	}
}
