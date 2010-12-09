#include <iostream>
#include <fstream>

using namespace std;

#include "neuralNet.h"
#include "chronometer.h"

#define PATH "/home/timon/test.nn"

#define INITIAL_WEIGHS_RANGE 20

void printTestParams(ImplementationType implementationType,
		VectorType vectorType, unsigned size)
{
	switch (implementationType)
	{
	case C:
		printf("    C     ");
		break;
	case SSE2:
		printf("   SSE2   ");
		break;
	case CUDA:
		printf("   CUDA   ");
		break;
	case CUDA2:
		printf("   CUDA2  ");
		break;
	case CUDA_INV:
		printf(" CUDA_INV ");
		break;
	}
	switch (vectorType)
	{
	case FLOAT:
		printf(" FLOAT ");
		break;
	case BIT:
		printf(" BIT   ");
		break;
	case SIGN:
		printf(" SIGN  ");
		break;
	case BYTE:
		printf(" BYTE  ");
		break;
	}
	printf(" size = %d weighsRange %d \n", size, INITIAL_WEIGHS_RANGE);
}

unsigned char areEqual(float expected, float actual, VectorType vectorType)
{
	if (vectorType == FLOAT)
	{
		return (expected - 1 < actual && expected + 1 > actual);
	}
	else
	{
		return expected == actual;
	}
}

unsigned assertEqualsInterfaces(Interface* expected, Interface* actual)
{
    if(expected->getVectorType() != actual->getVectorType()){
        throw "The interfaces are not even of the same type!";
    }
    if(expected->getSize() != actual->getSize()){
        throw "The interfaces are not even of the same size!";
    }
	unsigned differencesCounter = 0;

    for(unsigned i = 0;i < expected->getSize();i++){
        if(!areEqual(expected->getElement(i), actual->getElement(i), expected->getVectorType())){
            printf("The interfaces are not equal at the position %d (expected = %f actual %f).\n", i, expected->getElement(i), actual->getElement(i));
            ++differencesCounter;
        }
    }
	return differencesCounter;
}

#define SIZE_MIN 5
#define SIZE_MAX 5
#define SIZE_INC 5
#define NUM_LAYERS 2

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	//TODO AAA
	try
	{
//		for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++)
//		{
//			VectorType vectorType = (VectorType)vectType;
//			if (vectorType != BYTE)
//			{
//				for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
//						+= SIZE_INC)
//				{
//					NeuralNet controlNeuralNet(C);
//					controlNeuralNet.createFeedForwardNet(size, vectorType,
//							NUM_LAYERS, size, vectorType, IDENTITY);
////					controlNeuralNet.createFullyConnectedNet(size, vectorType,
////							NUM_LAYERS, size, vectorType, IDENTITY);
//					controlNeuralNet.randomWeighs(INITIAL_WEIGHS_RANGE);
//
//					FILE* stream = fopen(PATH, "w+b");
//					controlNeuralNet.save(stream);
//					fclose(stream);
//
//					controlNeuralNet.calculateOutput();
//
//					for (unsigned implType = 0; implType
//							< IMPLEMENTATION_TYPE_DIM; implType++)
//					{
//						ImplementationType implementationType =
//								(ImplementationType)((implType));
//						printTestParams(implementationType, vectorType, size);
//
//						NeuralNet nn(implementationType);
//						stream = fopen(PATH, "r+b");
//						nn.load(stream);
//						fclose(stream);
//						nn.getInput(0)->copyFrom(controlNeuralNet.getInput(0));
//
//						//test calculation
//						nn.calculateOutput();
//						unsigned differences =
//								assertEqualsInterfaces(controlNeuralNet.getOutput(0),
//										nn.getOutput(0));
//						if (differences != 0)
//							printf("Errors on outputs: %d \n", differences);
//					}
//				}
//			}
//		}
		printf("Exit success.\n");
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (std::string error)
	{
		cout << "Error: " << error << endl;
		//	} catch (...) {
		//		printf("An error was thrown.\n", 1);
	}

	//mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
