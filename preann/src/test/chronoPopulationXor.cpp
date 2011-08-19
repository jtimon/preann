#include <iostream>
#include <fstream>

using namespace std;

#include "taskXor.h"
#include "population.h"
#include "chronometer.h"

#define PATH "/home/timon/test.nn"

#define INITIAL_WEIGHS_RANGE 20

void printTestParams(ImplementationType implementationType,
		BufferType bufferType, unsigned size)
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
	switch (bufferType)
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

unsigned char areEqual(float expected, float actual, BufferType bufferType)
{
	if (bufferType == FLOAT)
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
	if (expected->getBufferType() != actual->getBufferType())
	{
		throw "The interfaces are not even of the same type!";
	}
	if (expected->getSize() != actual->getSize())
	{
		throw "The interfaces are not even of the same size!";
	}
	unsigned differencesCounter = 0;

	for (unsigned i = 0; i < expected->getSize(); i++)
	{
		if (!areEqual(expected->getElement(i), actual->getElement(i),
				expected->getBufferType()))
		{
			printf(
					"The interfaces are not equal at the position %d (expected = %f actual %f).\n",
					i, expected->getElement(i), actual->getElement(i));
			++differencesCounter;
		}
	}
	return differencesCounter;
}

#define SIZE_MIN 5
#define SIZE_MAX 5
#define SIZE_INC 5
#define NUM_TRIES 2

int main(int argc, char *argv[])
{/*
	Chronometer total;
	total.start();
	//TODO AAA
	try
	{
		for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size += SIZE_INC)
		{
			Task* xorTask = new TaskXor(size, NUM_TRIES);
			Individual example(SSE2);
			example.addInputLayer(size, BIT);
			example.addInputLayer(size, BIT);
			example.addLayer(size * 2, BIT, BINARY_STEP);
			example.addLayersConnection(0, 2);
			example.addLayersConnection(1, 2);
			example.addOutputLayer(size, BIT, BINARY_STEP);
			example.addLayersConnection(2, 3);

			Population pop(xorTask, &example, 100, 20);
			pop.setSelectionRanking(2, 5, 1);
			pop.setCrossoverUniformScheme(NEURON, 2, 0.9);
			pop.setMutationsPerIndividual(1, 0.5);

			while (pop.getBestIndividualScore() < 0 && pop.nextGeneration()< 1000){
				printf("Generation %d BestIndividualScore %f AverageScore %f\n", pop.getGeneration(), pop.getAverageScore(), pop.getBestIndividualScore());
			}
			delete (xorTask);
		}

		//		for (unsigned vectType = 0; vectType < BUFFER_TYPE_DIM; vectType++)
		//		{
		//			BufferType bufferType = (BufferType)vectType;
		//			if (bufferType != BYTE)
		//			{
		//				for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
		//						+= SIZE_INC)
		//				{
		//					NeuralNet controlNeuralNet(C);
		//					controlNeuralNet.createFeedForwardNet(size, bufferType,
		//							NUM_LAYERS, size, bufferType, IDENTITY);
		////					controlNeuralNet.createFullyConnectedNet(size, bufferType,
		////							NUM_LAYERS, size, bufferType, IDENTITY);
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
		//						printTestParams(implementationType, bufferType, size);
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
		MemoryManagement.printTotalAllocated();
		MemoryManagement.mem_printTotalPointers();
	} catch (std::string error)
	{
		cout << "Error: " << error << endl;
		//	} catch (...) {
		//		printf("An error was thrown.\n", 1);
	}

	//MemoryManagement.mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());*/
	return EXIT_SUCCESS;
}
