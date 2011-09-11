
#include <iostream>
#include <fstream>

using namespace std;

#include "test.h"
#include "population.h"
#include "chronometer.h"

unsigned memoryLosses = 0;
int size;
unsigned ouputSize = 100;

void checkAndPrintErrors(string testingClass, Test* test)
{
    if(MemoryManagement::getPtrCounter() > 0 || MemoryManagement::getTotalAllocated() > 0){
        cout << "Memory loss detected testing class " << testingClass << ".\n" << endl;
        test->printCurrentState();
        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
        memoryLosses++;
    }
}

void testBuffer(Test* test)
{
	Buffer* buffer = Factory::newBuffer(size, test->getBufferType(), test->getImplementationType());
	delete(buffer);

    checkAndPrintErrors("Buffer", test);
}

void testConnection(Test* test)
{
	Buffer* buffer = Factory::newBuffer(size, test->getBufferType(), test->getImplementationType());
	Connection* connection = Factory::newConnection(buffer, size, test->getImplementationType());

	delete(connection);
	delete(buffer);

    checkAndPrintErrors("Connection", test);
}

void testLayer(Test* test)
{
    Layer *layer = new Layer(size, test->getBufferType(), IDENTITY, test->getImplementationType());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    delete (layer);

    checkAndPrintErrors("Layer", test);
}

int main(int argc, char *argv[]) {

	Test test;
	Chronometer total;
	total.start();
	try {

		test.addIterator(&size, 100, 100, 100);
		test.withAll(ET_BUFFER);
		test.withAll(ET_IMPLEMENTATION);
		test.printParameters();

		test.testFunction(testBuffer, "Buffer::memory");

		test.exclude(ET_BUFFER, 1, BYTE);
		test.printParameters();

		test.testFunction(testConnection, "Connection::memory");
		test.testFunction(testLayer, "Layer::memory");

		printf("Exit success.\n", 1);
		MemoryManagement::printTotalAllocated();
		MemoryManagement::printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
	} catch (...) {
		printf("An error was thrown.\n", 1);
	}

	cout << "Total memory losses: " << memoryLosses << endl;
	MemoryManagement::printListOfPointers();

	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
