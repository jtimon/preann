
#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"

Chronometer total;
Test test;

unsigned testClone(Vector* toTest)
{
	Vector* copy = toTest->clone();
	unsigned differencesCounter = test.assertEquals(toTest, copy);

	if (toTest->getImplementationType() != copy->getImplementationType()){
		printf("The vectors are not of the same implementation type.\n");
		++differencesCounter;
	}

	delete(copy);
	return differencesCounter;
}

unsigned testCopyFrom(Vector* toTest)
{
	Interface interface = Interface(toTest->getSize(), toTest->getVectorType());
	interface.random(test.getInitialWeighsRange());

	unsigned differencesCounter = 0;
	Vector* cVector = Factory::newVector(toTest, C);

	toTest->copyFromInterface(&interface);
	cVector->copyFromInterface(&interface);

	differencesCounter += test.assertEquals(cVector, toTest);

	delete(cVector);
	return differencesCounter;
}

unsigned testCopyTo(Vector* toTest)
{
	Interface interface = Interface(toTest->getSize(), toTest->getVectorType());

	Vector* cVector = Factory::newVector(toTest, C);
	Interface cInterface = Interface(toTest->getSize(), toTest->getVectorType());

	toTest->copyToInterface(&interface);
	cVector->copyToInterface(&cInterface);

	unsigned differencesCounter = test.assertEqualsInterfaces(&cInterface, &interface);

	delete(cVector);
	return differencesCounter;
}

unsigned testActivation(Vector* toTest, FunctionType functionType)
{
	Vector* results = Factory::newVector(toTest->getSize(), FLOAT, toTest->getImplementationType());
	results->random(test.getInitialWeighsRange());

	Vector* cResults = Factory::newVector(results, C);
	Vector* cVector = Factory::newVector(toTest->getSize(), toTest->getVectorType(), C);

	toTest->activation(results, functionType);
	cVector->activation(cResults, functionType);
	unsigned differencesCounter = test.assertEquals(cVector, toTest);

	delete(results);
	delete(cVector);
	delete(cResults);
	return differencesCounter;
}

unsigned testAddToResults(Connection* toTest)
{
	unsigned outputSize = toTest->getSize() / toTest->getInput()->getSize();

	Vector* results = Factory::newVector(outputSize, FLOAT, toTest->getImplementationType());

	Vector* cInput = Factory::newVector(toTest->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(toTest);

	Vector* cResults = Factory::newVector(outputSize, FLOAT, C);

	toTest->calculateAndAddTo(results);
	cConnection->calculateAndAddTo(cResults);

	unsigned differencesCounter = test.assertEquals(cResults, results);

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
		float mutation = randomFloat(test.getInitialWeighsRange());
		unsigned pos = randomUnsigned(toTest->getSize());
		toTest->mutate(pos, mutation);
		cConnection->mutate(pos, mutation);
	}

	unsigned differences = test.assertEquals(cConnection, toTest);
	delete(cInput);
	delete(cConnection);
	return differences;
}

unsigned testCrossover(Connection* toTest)
{
	unsigned outputSize = toTest->getSize() / toTest->getInput()->getSize();

	Connection* other = Factory::newConnection(toTest->getInput(), outputSize, toTest->getImplementationType());
	other->random(test.getInitialWeighsRange());

	Vector* cInput = Factory::newVector(toTest->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(toTest);
	Connection* cOther = Factory::newConnection(cInput, outputSize, C);
	cOther->copyFrom(other);

	Interface bitVector = Interface(toTest->getSize(), BIT);
	bitVector.random(1);

	toTest->crossover(other, &bitVector);
	cConnection->crossover(cOther, &bitVector);

	unsigned differences = test.assertEquals(cConnection, toTest);
	differences += test.assertEquals(cOther, other);

	delete(other);
	delete(cInput);
	delete(cConnection);
	delete(cOther);
	return differences;
}

#define OUTPUT_SIZE_MIN 1
#define OUTPUT_SIZE_MAX 3
#define OUTPUT_SIZE_INC 2
#define NUM_MUTATIONS 10

#define PATH "/home/timon/layer.lay"

int main(int argc, char *argv[]) {

	unsigned errorCount = 0;
	total.start();

	test.setMinSize(2);
	test.setMaxSize(500);
	test.setIncSize(100);
	test.setInitialWeighsRange(20);
	test.printParameters();

	try {
		for (test.sizeToMin(); test.sizeIncrement(); ) {
			for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {
				for (test.implementationTypeToMin(); test.implementationTypeIncrement(); ) {
//					if (implementationType != C){
					{

					Vector* vector = Factory::newVector(test.getSize(), test.getVectorType(), test.getImplementationType());
					vector->random(test.getInitialWeighsRange());

					errorCount = testClone(vector);
				    if (errorCount != 0){
				    	test.printCurrentState();
				    	printf("Errors on clone: %d \n", errorCount);
				    }
					errorCount = testCopyTo(vector);
					if (errorCount != 0){
						test.printCurrentState();
						printf("Errors on copyTo: %d \n", errorCount);
					}
					errorCount = testCopyFrom(vector);
					if (errorCount != 0){
						test.printCurrentState();
						printf("Errors on copyTo: %d \n", errorCount);
					}
					if (test.getVectorType() != BYTE) {
						errorCount = testActivation(vector, IDENTITY);
						if (errorCount != 0){
							test.printCurrentState();
							printf("Errors on activation: %d \n", errorCount);
						}
					}
					if (test.getVectorType() != BYTE)
					for (unsigned outputSize = OUTPUT_SIZE_MIN; outputSize <= OUTPUT_SIZE_MAX; outputSize += OUTPUT_SIZE_INC) {

						Connection* connection = Factory::newConnection(vector, outputSize, test.getImplementationType());
						connection->random(test.getInitialWeighsRange());

						errorCount = testAddToResults(connection);
						if (errorCount != 0){
							test.printCurrentState();
							printf("Errors on Connection::addToResults: %d \n", errorCount);
						}
						errorCount = testMutate(connection, NUM_MUTATIONS);
						if (errorCount != 0){
							test.printCurrentState();
							printf("Errors on Connection::mutate: %d \n", errorCount);
						}
						errorCount = testCrossover(connection);
						if (errorCount != 0){
							test.printCurrentState();
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
