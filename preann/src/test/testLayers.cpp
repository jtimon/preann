
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"

#define PATH "/home/timon/layer.lay"

#define INITIAL_WEIGHS_RANGE 20

void printTestParams(ImplementationType implementationType, VectorType vectorType, unsigned size)
{
    switch (implementationType){
        case C: 		printf("    C     "); 	break;
        case SSE2: 		printf("   SSE2   ");	break;
        case CUDA: 		printf("   CUDA   ");	break;
        case CUDA2:		printf("   CUDA2  ");	break;
        case CUDA_INV:	printf(" CUDA_INV ");	break;
    }
    switch (vectorType){
        case FLOAT: printf(" FLOAT "); 	break;
        case BIT: 	printf(" BIT   ");	break;
        case SIGN: 	printf(" SIGN  ");	break;
        case BYTE:	printf(" BYTE  ");	break;
    }
    printf(" size = %d weighsRange %d \n", size, INITIAL_WEIGHS_RANGE);
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

#define NUM_INPUTS 3

Layer* createAndLoadLayer(ImplementationType implementationType, Vector** inputVectors)
{
    FILE* stream = fopen(PATH, "r+b");
    Layer* layer = new Layer(stream, implementationType);

    for (unsigned i = 0; i < NUM_INPUTS; i++){
		layer->addInput(inputVectors[i]);
	}
    layer->loadWeighs(stream);
    fclose(stream);
    return layer;
}

Layer* createAndSaveLayer(unsigned& size, VectorType vectorType, Vector** controlInputs)
{
    Layer* controlLayer = new Layer(size, vectorType, IDENTITY, C);

    for (unsigned i = 0; i < NUM_INPUTS; i++){
		controlLayer->addInput(controlInputs[i]);
	}
    controlLayer->randomWeighs(INITIAL_WEIGHS_RANGE);

	FILE* stream = fopen(PATH, "w+b");
	controlLayer->save(stream);
	controlLayer->saveWeighs(stream);
	fclose(stream);
    return controlLayer;
}

void testLayer(unsigned size, VectorType vectorType)
{
	Vector* controlInputVectors[VECTOR_TYPE_DIM];
	for (unsigned i = 0; i < NUM_INPUTS; i++) {
		VectorType vectorTypeAux = BYTE;
		while (vectorTypeAux == BYTE) {
			vectorTypeAux = (VectorType)randomUnsigned(VECTOR_TYPE_DIM);
		}
		controlInputVectors[i] = Factory::newVector(size, vectorTypeAux, C);
		controlInputVectors[i]->random(INITIAL_WEIGHS_RANGE);
	}

    Layer* controlLayer = createAndSaveLayer(size, vectorType, controlInputVectors);
    controlLayer->calculateOutput();

    for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {
		ImplementationType implementationType = (ImplementationType)((implType));

		printTestParams(implementationType, vectorType, size);

		Vector* inputVectors[VECTOR_TYPE_DIM];
		for (unsigned i = 0; i < NUM_INPUTS; i++) {
			inputVectors[i] = Factory::newVector(size, controlInputVectors[i]->getVectorType(), implementationType);
			inputVectors[i]->copyFrom(controlInputVectors[i]);
		}

		Layer* layer = createAndLoadLayer(implementationType, inputVectors);

	    //test calculation
		layer->calculateOutput();

	    unsigned differences = assertEquals(controlLayer->getOutput(), layer->getOutput());
	    if (differences != 0)
	    	printf("Errors on outputs: %d \n", differences);

		//test Weighs
	    for(unsigned i = 0; i < NUM_INPUTS; i++){
	        Connection* expectedWeighs = controlLayer->getConnection(i);
	        Connection* actualWeighs = layer->getConnection(i);
	        differences = assertEquals(expectedWeighs, actualWeighs);
	        if (differences != 0)
	        	printf("Errors on weighs (input %d): %d \n", i, differences);
	    }
		delete (layer);
		for (unsigned i = 0; i < NUM_INPUTS; i++) {
			delete(inputVectors[i]);
		}
	}
    delete (controlLayer);
	for (unsigned i = 0; i < NUM_INPUTS; i++) {
		delete(controlInputVectors[i]);
	}
}

#define SIZE_MIN 1
#define SIZE_MAX 50
#define SIZE_INC 50

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();

	try {
		for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size += SIZE_INC) {
			for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
				if (vectType != BYTE)
					testLayer(size, (VectorType) vectType);
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
