
#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "factory.h"

#define INITIAL_WEIGHS_RANGE 20

void printTestParams(ImplementationType implementationType, VectorType vectorType, unsigned size, unsigned weighsRange)
{
    switch (implementationType){
        case C: 		printf(" C        "); 	break;
        case SSE2: 		printf(" SSE2     ");	break;
        case CUDA: 		printf(" CUDA     ");	break;
        case CUDA2:		printf(" CUDA2    ");	break;
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

	toTest->copyFromInterface(interface);
	cVector->copyFromInterface(interface);

	differencesCounter += assertEquals(cVector, toTest);

	delete(cVector);
	delete(interface);
	return differencesCounter;
}

unsigned testCopyTo(Vector* toTest)
{
	Interface* interface = new Interface(toTest->getSize(), toTest->getVectorType());

	Vector* cVector = Factory::newVector(toTest, C);
	Interface* cInterface = new Interface(toTest->getSize(), toTest->getVectorType());

	toTest->copyToInterface(interface);
	cVector->copyToInterface(cInterface);

	unsigned differencesCounter = assertEqualsInterfaces(cInterface, interface);

	delete(interface);
	delete(cVector);
	delete(cInterface);
	return differencesCounter;
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

	delete(results);
	delete(cVector);
	delete(cResults);
	return differencesCounter;
}

unsigned testMutate(Vector* toTest, unsigned times)
{
	Vector* cVector = Factory::newVector(toTest, C);

	for(unsigned i=0; i < times; i++) {
		float mutation = randomFloat(INITIAL_WEIGHS_RANGE);
		unsigned pos = randomUnsigned(toTest->getSize());
		toTest->mutate(pos, mutation);
		cVector->mutate(pos, mutation);
	}

	unsigned differences = assertEquals(cVector, toTest);
	delete(cVector);
	return differences;
}

unsigned testCrossover(Vector* toTest)
{
	Interface* bitVector = new Interface(toTest->getSize(), BIT);
	bitVector->random(1);

	Vector* other = Factory::newVector(toTest->getSize(), toTest->getVectorType(), toTest->getImplementationType());
	other->random(INITIAL_WEIGHS_RANGE);

	Vector* cVector = Factory::newVector(toTest, C);
	Vector* cOther = Factory::newVector(other, C);

	toTest->crossover(other, bitVector);
	cVector->crossover(cOther, bitVector);

	unsigned differences = assertEquals(cVector, toTest);
	differences += assertEquals(cOther, other);

	delete(bitVector);
	delete(other);
	delete(cVector);
	delete(cOther);
	return differences;
}

unsigned testAddToResults(Connection* toTest)
{
	unsigned outputSize = toTest->getSize() / toTest->getInput()->getSize();

	Vector* results = Factory::newVector(outputSize, FLOAT, toTest->getImplementationType());

	Vector* cInput = Factory::newVector(toTest->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(toTest);

	Vector* cResults = Factory::newVector(outputSize, FLOAT, C);

	toTest->addToResults(results);
	cConnection->addToResults(cResults);

	unsigned differencesCounter = assertEquals(cResults, results);

	delete(results);
	delete(cInput);
	delete(cConnection);
	delete(cResults);
	return differencesCounter;
}

unsigned testMutate(Connection* toTest, unsigned times)
{
	unsigned outputSize = toTest->getSize() / toTest->getInput()->getSize();

	Vector* cInput = Factory::newVector(toTest->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(toTest);

	for(unsigned i=0; i < times; i++) {
		float mutation = randomFloat(INITIAL_WEIGHS_RANGE);
		unsigned pos = randomUnsigned(toTest->getSize());
		toTest->mutate(pos, mutation);
		cConnection->mutate(pos, mutation);
	}

	unsigned differences = assertEquals(cConnection, toTest);
	delete(cInput);
	delete(cConnection);
	return differences;
}

unsigned testCrossover(Connection* toTest)
{
	unsigned outputSize = toTest->getSize() / toTest->getInput()->getSize();

	Connection* other = Factory::newConnection(toTest->getInput(), outputSize, toTest->getImplementationType());
	other->random(INITIAL_WEIGHS_RANGE);

	Vector* cInput = Factory::newVector(toTest->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(toTest);
	Connection* cOther = Factory::newConnection(cInput, outputSize, C);
	cOther->copyFrom(other);

	Interface* bitVector = new Interface(toTest->getSize(), BIT);
	bitVector->random(1);

	toTest->crossover(other, bitVector);
	cConnection->crossover(cOther, bitVector);

	unsigned differences = assertEquals(cConnection, toTest);
	differences += assertEquals(cOther, other);

	delete(bitVector);
	delete(other);
	delete(cInput);
	delete(cConnection);
	delete(cOther);
	return differences;
}

#define SIZE_MIN 2
#define SIZE_MAX 128
#define SIZE_INC 16
#define OUTPUT_SIZE_MIN 3
#define OUTPUT_SIZE_MAX 3
#define OUTPUT_SIZE_INC 3
#define NUM_MUTATIONS 10

#define PATH "/home/timon/layer.lay"

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();
	unsigned errorCount = 0;

	try {
		for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size += SIZE_INC) {
			for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
				VectorType vectorType = (VectorType)((vectType));
				FunctionType functionType = (FunctionType)(vectType);

				for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {
					ImplementationType implementationType = (ImplementationType)((implType));
//					if (implementationType != C){
					{
					printf("----------------------------\n");
					printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);

					Vector* vector = Factory::newVector(size, (VectorType)vectType, implementationType);
					vector->random(INITIAL_WEIGHS_RANGE);

					errorCount = testClone(vector);
				    if (errorCount != 0){
				    	printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
				    	printf("Errors on clone: %d \n", errorCount);
				    }
					errorCount = testCopyTo(vector);
					if (errorCount != 0){
						printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
						printf("Errors on copyTo: %d \n", errorCount);
					}
					errorCount = testCopyFrom(vector);
					if (errorCount != 0){
						printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
						printf("Errors on copyTo: %d \n", errorCount);
					}
					if (vectorType != BYTE) {
						errorCount = testActivation(vector, functionType);
						if (errorCount != 0){
							printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
							printf("Errors on activation: %d \n", errorCount);
						}
					}
					if (vectorType != BIT && vectorType != SIGN) {
						errorCount = testMutate(vector, NUM_MUTATIONS);
						if (errorCount != 0){
							printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
							printf("Errors on mutate: %d \n", errorCount);
						}
						errorCount = testCrossover(vector);
						if (errorCount != 0){
							printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
							printf("Errors on crossover: %d \n", errorCount);
						}
					}
					if (vectorType != BYTE)
					for (unsigned outputSize = OUTPUT_SIZE_MIN; outputSize <= OUTPUT_SIZE_MAX; outputSize += OUTPUT_SIZE_INC) {

						Connection* connection = Factory::newConnection(vector, outputSize, implementationType);
						connection->random(INITIAL_WEIGHS_RANGE);

						errorCount = testAddToResults(connection);
						if (errorCount != 0){
							printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
							printf("Errors on Connection::addToResults: %d \n", errorCount);
						}
						errorCount = testMutate(connection, NUM_MUTATIONS);
						if (errorCount != 0){
							printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
							printf("Errors on Connection::mutate: %d \n", errorCount);
						}
						errorCount = testCrossover(connection);
						if (errorCount != 0){
							printTestParams(implementationType, vectorType, size, INITIAL_WEIGHS_RANGE);
							printf("Errors on Connection::crossover: %d \n", errorCount);
						}
						delete(connection);
					}
					delete(vector);
					}
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
