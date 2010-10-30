
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "cudaVector2.h"

#define PATH "/home/timon/layer.lay"

#define INITIAL_WEIGHS_RANGE 20


void printTestParams(ImplementationType implementationType, VectorType vectorType, unsigned size, unsigned numInputs)
{
    switch (implementationType){
        case C: 	printf(" C     "); 	break;
        case SSE2: 	printf(" SSE2  ");	break;
        case CUDA: 	printf(" CUDA  ");	break;
        case CUDA2:	printf(" CUDA2 ");	break;
    }
    switch (vectorType){
        case FLOAT: printf(" FLOAT "); 	break;
        case BIT: 	printf(" BIT   ");	break;
        case SIGN: 	printf(" SIGN  ");	break;
        case BYTE:	printf(" BYTE  ");	break;
    }
    printf(" size = %d numInputs %d weighsRange %d \n", size, numInputs, INITIAL_WEIGHS_RANGE);
}

unsigned char areEqual(float expected, float actual, VectorType vectorType)
{
	if (vectorType == FLOAT){
		return (expected - 1 < actual
			 && expected + 1 > actual);
	} else {
		return expected == actual;
	}
}

unsigned assertEquals(Vector* expected, Vector* actual)
{
    if(expected->getVectorType() != actual->getVectorType()){
        throw "The vectors are not even of the same type!";
    }
    if(expected->getSize() != actual->getSize()){
        throw "The vectors are not even of the same size!";
    }

	unsigned differencesCounter = 0;
	Interface* expectedInt = expected->toInterface();
	Interface* actualInt = actual->toInterface();

    for(unsigned i = 0;i < expectedInt->getSize();i++){
        if(!areEqual(expectedInt->getElement(i), actualInt->getElement(i), expectedInt->getVectorType())){
            printf("The vectors are not equal at the position %d (expected = %f actual %f).\n", i, expectedInt->getElement(i), actualInt->getElement(i));
            ++differencesCounter;
        }
    }
    delete(expectedInt);
	delete(actualInt);
	return differencesCounter;
}

Layer* createAndLoadLayer(ImplementationType implementationType, Vector* controlInput, unsigned numInputs)
{
    FILE* stream = fopen(PATH, "r+b");
    Layer *layer = new Layer(stream, implementationType);
    fclose(stream);

    for (unsigned i = 0; i < numInputs; i++){
		layer->setInput(controlInput, i);
	}
    return layer;
}

Layer* createAndSaveLayer(unsigned& size, VectorType vectorType, Vector* controlInput, unsigned numInputs)
{
    Layer* controlLayer = new Layer(size, vectorType, IDENTITY, C);

    for (unsigned i = 0; i < numInputs; i++){
		controlLayer->addInput(controlInput);
	}
    controlLayer->randomWeighs(INITIAL_WEIGHS_RANGE);

	FILE* stream = fopen(PATH, "w+b");
	controlLayer->save(stream);
	fclose(stream);
    return controlLayer;
}

#define IMPLEMENTATION_TYPE_DIM 4

void testLayer(unsigned size, VectorType vectorType, unsigned numInputs)
{
    Vector *controlInputVector = Factory::newVector(size, vectorType, C);
    controlInputVector->random(INITIAL_WEIGHS_RANGE);

    Layer* controlLayer = createAndSaveLayer(size, vectorType, controlInputVector, numInputs);
    controlLayer->calculateOutput();

    for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {
		ImplementationType implementationType = (ImplementationType)((implType));

		printTestParams(implementationType, vectorType, size, numInputs);

		Vector* inputVector = Factory::newVector(size, vectorType, implementationType);
		inputVector->copyFromVector(controlInputVector);

		Layer* layer = createAndLoadLayer(implementationType, inputVector, numInputs);

	    //test calculation
		layer->calculateOutput();

	    unsigned differences = assertEquals(controlLayer->getOutput(), layer->getOutput());
	    if (differences != 0)
	    	printf("Errors on outputs: %d \n", differences);


		//test Weighs
	    for(unsigned i = 0; i < numInputs; i++){
	        Vector* expectedWeighs = controlLayer->getConnection(i);
	        Vector* actualWeighs = layer->getConnection(i);
	        if(implementationType == CUDA2){
	            unsigned inputSize = actualWeighs->getSize() / layer->getOutput()->getSize();
	            actualWeighs->transposeMatrix(inputSize);
	        }
	        differences = assertEquals(expectedWeighs, actualWeighs);
	        if (differences != 0)
	        	printf("Errors on weighs (input %d): %d \n", i, differences);
	    }

		delete (layer);
	    delete (inputVector);
	}
    delete (controlLayer);
    delete (controlInputVector);
}

#define VECTOR_TYPE_DIM 2
#define SIZE_MAX 100
#define SIZE_INC 10
#define NUM_INPUTS 2

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();

	try {
		for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
//			if (vectType != BYTE)
			if (vectType == SIGN)
				for (unsigned size = 1; size < SIZE_MAX; size += SIZE_INC) {
					testLayer(size, (VectorType) vectType, NUM_INPUTS);
				}
		}

		printf("Exit success.\n");
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
//	} catch (...) {
//		printf("An error was thrown.\n", 1);
	}

	//mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
