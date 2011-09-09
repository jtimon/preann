
#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"

unsigned testActivation(Test* test)
{
	Buffer* buffer = test->buildBuffer();
	FunctionType functionType = test->getFunctionType();
	unsigned differencesCounter = 0;

	Buffer* results = Factory::newBuffer(buffer->getSize(), FLOAT, buffer->getImplementationType());
	results->random(test->getInitialWeighsRange());

	Buffer* cResults = Factory::newBuffer(results, C);
	Buffer* cBuffer = Factory::newBuffer(buffer->getSize(), buffer->getBufferType(), C);

	buffer->activation(results, functionType);
	cBuffer->activation(cResults, functionType);
	differencesCounter += Test::assertEquals(cBuffer, buffer);

	delete(results);
	delete(buffer);
	delete(cBuffer);
	delete(cResults);

	return differencesCounter;
}

unsigned testCopyFromInterface(Test* test)
{
	Buffer* buffer = test->buildBuffer();
	unsigned differencesCounter = 0;

	Interface interface(buffer->getSize(), buffer->getBufferType());
	interface.random(test->getInitialWeighsRange());

	Buffer* cBuffer = Factory::newBuffer(buffer, C);

	buffer->copyFromInterface(&interface);
	cBuffer->copyFromInterface(&interface);

	differencesCounter += Test::assertEquals(cBuffer, buffer);

	delete(cBuffer);
	delete(buffer);

	return differencesCounter;
}

unsigned testCopyToInterface(Test* test)
{
	Buffer* buffer = test->buildBuffer();
	unsigned differencesCounter = 0;

	Interface interface = Interface(buffer->getSize(), buffer->getBufferType());

	Buffer* cBuffer = Factory::newBuffer(buffer, C);
	Interface cInterface = Interface(buffer->getSize(), buffer->getBufferType());

	buffer->copyToInterface(&interface);
	cBuffer->copyToInterface(&cInterface);

	differencesCounter = Test::assertEqualsInterfaces(&cInterface, &interface);

	delete(cBuffer);
	delete(buffer);

	return differencesCounter;
}

unsigned testClone(Test* test)
{
	Buffer* buffer = test->buildBuffer();
	unsigned differencesCounter = 0;
	Buffer* copy = buffer->clone();
	differencesCounter += Test::assertEquals(buffer, copy);
	delete(copy);
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
	test.printParameters();

	try {
		test.test(testClone, "Buffer::clone");
		test.test(testCopyFromInterface, "Buffer::copyFromInterface");
		test.test(testCopyToInterface, "Buffer::copyToInterface");
		test.disableBufferType(BYTE);
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
