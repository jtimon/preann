
#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"

unsigned testCalculateAndAddTo(Test* test)
{
	Buffer* buffer = test->buildBuffer();
	Connection* connection = test->buildConnection(buffer);
	unsigned differencesCounter = 0;

	Buffer* results = Factory::newBuffer(test->getOutputSize(), FLOAT, connection->getImplementationType());

	Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, test->getOutputSize(), C);
	cConnection->copyFrom(connection);

	Buffer* cResults = Factory::newBuffer(test->getOutputSize(), FLOAT, C);

	connection->calculateAndAddTo(results);
	cConnection->calculateAndAddTo(cResults);

	differencesCounter = Test::assertEquals(cResults, results);

	delete(results);
	delete(cInput);
	delete(cConnection);
	delete(cResults);

	delete(connection);
	delete(buffer);
	return differencesCounter;
}

unsigned testMutate(Test* test)
{
	Buffer* buffer = test->buildBuffer();
	Connection* connection = test->buildConnection(buffer);
	unsigned differencesCounter = 0;

	unsigned pos = Random::positiveInteger(connection->getSize());
	float mutation = Random::floatNum(test->getInitialWeighsRange());
	connection->mutate(pos, mutation);

	Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, test->getOutputSize(), C);
	cConnection->copyFrom(connection);

	for(unsigned i=0; i < NUM_MUTATIONS; i++) {
		float mutation = Random::floatNum(test->getInitialWeighsRange());
		unsigned pos = Random::positiveInteger(connection->getSize());
		connection->mutate(pos, mutation);
		cConnection->mutate(pos, mutation);
	}

	differencesCounter = Test::assertEquals(cConnection, connection);
	delete(cInput);
	delete(cConnection);

	delete(connection);
	delete(buffer);
	return differencesCounter;
}

unsigned testCrossover(Test* test)
{
	Buffer* buffer = test->buildBuffer();
	Connection* connection = test->buildConnection(buffer);
	unsigned differencesCounter = 0;

	Connection* other = Factory::newConnection(connection->getInput(), test->getOutputSize(), connection->getImplementationType());
	other->random(test->getInitialWeighsRange());

	Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
	Connection* cConnection = Factory::newConnection(cInput, test->getOutputSize(), C);
	cConnection->copyFrom(connection);
	Connection* cOther = Factory::newConnection(cInput, test->getOutputSize(), C);
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

	delete(connection);
	delete(buffer);
	return differencesCounter;
}

int main(int argc, char *argv[]) {

	Chronometer total;
	total.start();

	Test test;

	test.fromToBySize(2, 10, 10);
	test.fromToByOutputSize(1, 3, 2);
	test.setInitialWeighsRange(20);
	test.exclude(ET_BUFFER, 1, BYTE);
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
