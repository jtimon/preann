
#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"

int size;
float initialWeighsRange = 20;

unsigned testActivation(Test* test)
{
	START_BUFFER_TEST

	FunctionType functionType = test->getFunctionType();
	Buffer* results = Factory::newBuffer(buffer->getSize(), FLOAT, buffer->getImplementationType());
	results->random(initialWeighsRange);

	Buffer* cResults = Factory::newBuffer(results, C);
	Buffer* cBuffer = Factory::newBuffer(buffer->getSize(), buffer->getBufferType(), C);

	buffer->activation(results, functionType);
	cBuffer->activation(cResults, functionType);
	differencesCounter += Test::assertEquals(cBuffer, buffer);

	delete(results);
	delete(cBuffer);
	delete(cResults);

	END_BUFFER_TEST
}

unsigned testCopyFromInterface(Test* test)
{
	START_BUFFER_TEST

	Interface interface(buffer->getSize(), buffer->getBufferType());
	interface.random(initialWeighsRange);

	Buffer* cBuffer = Factory::newBuffer(buffer, C);

	buffer->copyFromInterface(&interface);
	cBuffer->copyFromInterface(&interface);

	differencesCounter += Test::assertEquals(cBuffer, buffer);

	delete(cBuffer);

	END_BUFFER_TEST
}

unsigned testCopyToInterface(Test* test)
{
	START_BUFFER_TEST

	Interface interface = Interface(buffer->getSize(), buffer->getBufferType());

	Buffer* cBuffer = Factory::newBuffer(buffer, C);
	Interface cInterface = Interface(buffer->getSize(), buffer->getBufferType());

	buffer->copyToInterface(&interface);
	cBuffer->copyToInterface(&cInterface);

	differencesCounter = Test::assertEqualsInterfaces(&cInterface, &interface);

	delete(cBuffer);

	END_BUFFER_TEST
}

unsigned testClone(Test* test)
{
//	START_BUFFER_TEST
	START_TEST
	Buffer* buffer = Factory::newBuffer(size, test->getBufferType(), test->getImplementationType());
	buffer->random(initialWeighsRange);

	Buffer* copy = buffer->clone();
	differencesCounter += Test::assertEquals(buffer, copy);
	delete(copy);

	END_BUFFER_TEST
}

int main(int argc, char *argv[]) {

	Chronometer total;
	total.start();

	Test test;
	test.with(ET_BUFFER, 1, BIT);
	test.withAll(ET_IMPLEMENTATION);
	test.addIterator(&size, 10, 10, 10);
	test.printParameters();

	try {
		test.test(testClone, "Buffer::clone");
		test.test(testCopyFromInterface, "Buffer::copyFromInterface");
		test.test(testCopyToInterface, "Buffer::copyToInterface");

		test.exclude(ET_BUFFER, 1, BYTE);
		test.test(testActivation, "Buffer::activation");

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
