
#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"

#define NUM_MUTATIONS 10

int size;
int outputSize = 100;
float initialWeighsRange = 20;

unsigned testCalculateAndAddTo(Test* test)
{
	START_CONNECTION_TEST

	Buffer* results = Factory::newBuffer(outputSize, FLOAT, connection->getImplementationType());

	Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(connection);

	Buffer* cResults = Factory::newBuffer(outputSize, FLOAT, C);

	connection->calculateAndAddTo(results);
	cConnection->calculateAndAddTo(cResults);

	differencesCounter = Test::assertEquals(cResults, results);

	delete(results);
	delete(cInput);
	delete(cConnection);
	delete(cResults);

	END_CONNECTION_TEST
}

unsigned testMutate(Test* test)
{
	START_CONNECTION_TEST

	unsigned pos = Random::positiveInteger(connection->getSize());
	float mutation = Random::floatNum(initialWeighsRange);
	connection->mutate(pos, mutation);

	Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(connection);

	for(unsigned i=0; i < NUM_MUTATIONS; i++) {
		float mutation = Random::floatNum(initialWeighsRange);
		unsigned pos = Random::positiveInteger(connection->getSize());
		connection->mutate(pos, mutation);
		cConnection->mutate(pos, mutation);
	}

	differencesCounter = Test::assertEquals(cConnection, connection);
	delete(cInput);
	delete(cConnection);

	END_CONNECTION_TEST
}

unsigned testCrossover(Test* test)
{
	START_CONNECTION_TEST

	Connection* other = Factory::newConnection(connection->getInput(), outputSize, connection->getImplementationType());
	other->random(initialWeighsRange);

	Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
	cConnection->copyFrom(connection);
	Connection* cOther = Factory::newConnection(cInput, outputSize, C);
	cOther->copyFrom(other);

	Interface bitBuffer = Interface(connection->getSize(), BIT);
	//TODO bitBuffer.random(2); ??
	bitBuffer.random(1);

	connection->crossover(other, &bitBuffer);
	cConnection->crossover(cOther, &bitBuffer);

	differencesCounter = Test::assertEquals(cConnection, connection);
	differencesCounter += Test::assertEquals(cOther, other);

	delete(other);
	delete(cInput);
	delete(cConnection);
	delete(cOther);

	END_CONNECTION_TEST
}

int main(int argc, char *argv[]) {

	Chronometer total;
	total.start();

	Test test;

	test.addIterator(&size, 2, 12, 10);
	test.addIterator(&outputSize, 1, 3, 2);
	test.exclude(ET_BUFFER, 1, BYTE);
	test.withAll(ET_IMPLEMENTATION);
	test.printParameters();

	try {
		test.test(testCalculateAndAddTo, "Connection::calculateAndAddTo");
		test.test(testMutate, "Connection::mutate");
		test.test(testCrossover, "Connection::crossover");

		printf("Exit success.\n");
		MemoryManagement::printTotalAllocated();
		MemoryManagement::printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
//	} catch (...) {
//		printf("An error was thrown.\n", 1);
	}

	//MemoryManagement::mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
