
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "cudaLayer2.h"

#define PATH "/home/timon/layer.lay"

#define VECTOR_TYPE_DIM 2
#define IMPLEMENTATION_TYPE_DIM 4
#define INITIAL_WEIGHS_RANGE 20

#define MAX_SIZE 200
#define NUM_INPUTS 2

void printImplementationType(ImplementationType implementationType)
{
    switch (implementationType){
        case C:
            printf(" C ");
            break;
        case SSE2:
            printf(" SSE2 ");
            break;
        case CUDA:
            printf(" CUDA ");
            break;
        case CUDA2:
            printf(" CUDA2 ");
            break;
    }
}

void printVectorType(VectorType vectorType)
{
    switch (vectorType){
        case FLOAT:
            printf(" FLOAT ");
            break;
        case BIT:
            printf(" BIT ");
            break;
        case SIGN:
            printf(" SIGN ");
            break;
    }
}

void printTestParams(ImplementationType implementationType, VectorType vectorType, unsigned size)
{
    printImplementationType(implementationType);
    printVectorType(vectorType);
    printf(" size = %d numInputs %d weighsRange %d \n", size, NUM_INPUTS, INITIAL_WEIGHS_RANGE);
}

void assertEquals(Interface* expected, Interface* actual)
{
	if (expected->getVectorType() != actual->getVectorType()){
		throw "The interfaces are not even of the same type!";
	}
	if (expected->getSize() != actual->getSize()){
		throw "The interfaces are not even of the same size!";
	}
	for (unsigned i=0; i < expected->getSize(); i++) {
		if ((int)expected->getElement(i) != (int)actual->getElement(i)){
			printf("expected:\n");
			expected->print();
			printf("actual:\n");
			actual->print();
			char buffer[100];
			sprintf(buffer, "The interfaces are not equal at the position %d.", i);
			std::string error = buffer;
			throw error;
		}
	}
}

Layer* createAndLoadLayer(ImplementationType implementationType, FILE* stream, Interface* controlInput)
{
    Layer *layer = Factory::newLayer(implementationType);
    stream = fopen(PATH, "r+b");
    layer->load(stream);
    fclose(stream);
    for (unsigned numInputs = 0; numInputs < NUM_INPUTS; numInputs++){
					layer->setInput(controlInput, numInputs);
				}
    return layer;
}

Layer* createAndSaveLayer(unsigned& size, VectorType& vectorType, Interface* controlInput)
{
    Layer *controlLayer = Factory::newLayer(size, vectorType, C, IDENTITY);
    for (unsigned numInputs = 0; numInputs < NUM_INPUTS; numInputs++){
				controlLayer->addInput(controlInput);
			}
    controlLayer->randomWeighs(INITIAL_WEIGHS_RANGE);

	FILE* stream = fopen(PATH, "w+b");
	controlLayer->save(stream);
	fclose(stream);
    return controlLayer;
}

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();

	unsigned errorCount = 0;

	try {
		for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
			VectorType vectorType = (VectorType) vectType;
			printVectorType(vectorType);
			printf("\n---------\n");
		if (vectType != BYTE) for (unsigned size = 1; size < MAX_SIZE; size++){

			Interface* controlInput = new Interface(size, vectorType);
			controlInput->random(INITIAL_WEIGHS_RANGE);

			Layer* controlLayer = createAndSaveLayer(size, vectorType, controlInput);

			controlLayer->calculateOutput();
			Interface* expectedOutput = controlLayer->getOutput()->toInterface();

			for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {

				ImplementationType implementationType = (ImplementationType)(implType);

				Layer* layer = createAndLoadLayer(implementationType, stream, controlInput);
				layer->calculateOutput();
				Interface* actual = layer->getOutput()->toInterface();

				try{
					assertEquals(expectedOutput, actual);
				} catch (std::string error) {
					printTestParams(implementationType, vectorType, size);
					cout << ++errorCount <<" Error on outputs: " << error <<endl<<endl;
				}
				delete (actual);

				for (unsigned i = 0; i < NUM_INPUTS; i++){
					Interface* expectedWeighs = controlLayer->getConnection(i)->toInterface();
					Interface* actualWeighs = layer->getConnection(i)->toInterface();

					if(implementationType == CUDA2){
						unsigned inputSize = actualWeighs->getSize() / layer->getOutput()->getSize();
						actualWeighs->transposeMatrix(inputSize);
					}

					try{
						assertEquals(expectedWeighs, actualWeighs);
					} catch (std::string error) {
						printTestParams(implementationType, vectorType, size);
						cout << ++errorCount <<" Error on weighs: " << error <<endl<<endl;
					}
					delete(expectedWeighs);
					delete(actualWeighs);
				}
				delete(layer);
			}
			delete (expectedOutput);
			delete(controlLayer);
		}}

		if (errorCount == 0) {
			printf("Exit success.\n");
		} else {
			printf("Exit with %d errors.\n", errorCount);
		}
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
	} catch (...) {
		printf("An error was thrown.\n", 1);
	}

	//mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
