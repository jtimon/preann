
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "cudaVector.h"

#define INITIAL_WEIGHS_RANGE 20

void printTestParams(ImplementationType implementationType, VectorType vectorType, unsigned size, unsigned weighsRange)
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

unsigned testClone(Vector* toTest)
{
	Vector* copy = toTest->clone();
	unsigned differencesCounter = assertEquals(toTest, copy);

	if (toTest->getImplementationType() != copy->getImplementationType()){
		printf("The vectors are not of the same implementation type.\n");
		++differencesCounter;
	}

	delete(copy);
	return differencesCounter;
}

unsigned testCopyFrom(Vector* toTest)
{
	Interface* interface = new Interface(toTest->getSize(), toTest->getVectorType());
	interface->random(INITIAL_WEIGHS_RANGE);

	unsigned differencesCounter = 0;
	Vector* cVector = Factory::newVector(toTest, C);

	toTest->copyFrom(interface);
	cVector->copyFrom(interface);

	differencesCounter += assertEquals(cVector, toTest);

	delete(cVector);
	delete(interface);
	return differencesCounter;
}

unsigned testCopyTo(Vector* toTest)
{
	Vector* cVector = Factory::newVector(toTest, C);

	unsigned differencesCounter = 0;
	Interface* iA = new Interface(toTest->getSize(), toTest->getVectorType());
	Interface* iB = new Interface(toTest->getSize(), toTest->getVectorType());

	toTest->copyTo(iA);
	cVector->copyTo(iB);

	differencesCounter += assertEqualsInterfaces(iB, iA);

	delete(iA);
	delete(iB);
	delete(cVector);
	return differencesCounter;
}

unsigned testInputCalculation(Vector* input, Vector* inputWeighs)
{

}

unsigned testActivation(Vector* toTest, FunctionType functionType)
{
	Vector* results = Factory::newVector(toTest->getSize(), FLOAT, toTest->getImplementationType());
	results->random(INITIAL_WEIGHS_RANGE);

	Vector* cResults = Factory::newVector(results, C);
	Vector* cVector = Factory::newVector(toTest->getSize(), toTest->getVectorType(), C);

	toTest->activation(results, functionType);
	cVector->activation(cResults, functionType);
	unsigned differencesCounter = assertEquals(cVector, toTest);

	delete(cVector);
	return differencesCounter;
}

unsigned testMutate(Vector* toTest, unsigned pos, float mutation, unsigned inputSize)
{
	Vector* cVector = Factory::newVector(toTest, C);

	toTest->mutate(pos, mutation, inputSize);
	cVector->mutate(pos, mutation, inputSize);

	unsigned differencesCounter = assertEquals(cVector, toTest);

	delete(cVector);
	return differencesCounter;
}

unsigned testWeighCrossover(Vector* other, Interface* bitVector, unsigned inputSize)
{

}

#define IMPLEMENTATION_TYPE_DIM 4
#define VECTOR_TYPE_DIM 3
#define SIZE_MAX 5
#define SIZE_INC 1
#define PATH "/home/timon/layer.lay"

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();
	unsigned errorCount = 0;

	try {

		for (unsigned size = 1; size < SIZE_MAX; size += SIZE_INC) {
			for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
				VectorType vectorType = (VectorType)((vectType));
				FunctionType functionType = (FunctionType)(vectType);

				for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {
					ImplementationType implementationType = (ImplementationType)((implType));

					Vector* vector = Factory::newVector(size, (VectorType)vectType, implementationType);
					vector->random(INITIAL_WEIGHS_RANGE);

					errorCount += testClone(vector);
				    if (errorCount != 0){
				    	printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
				    	printf("Errors on clone: %d \n", errorCount);
				    }
					errorCount += testCopyTo(vector);
					if (errorCount != 0){
						printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
						printf("Errors on copyTo: %d \n", errorCount);
					}
					errorCount += testCopyFrom(vector);
					if (errorCount != 0){
						printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
						printf("Errors on copyTo: %d \n", errorCount);
					}
					errorCount += testActivation(vector, functionType);
					if (errorCount != 0){
						printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
						printf("Errors on activation: %d \n", errorCount);
					}

					delete(vector);
				}
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
