
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
	Buffer* buffer = Factory::newBuffer(size, (BufferType)test->getEnum(ET_BUFFER), (ImplementationType)test->getEnum(ET_IMPLEMENTATION));
	delete(buffer);

    checkAndPrintErrors("Buffer", test);
}

void testConnection(Test* test)
{
	Buffer* buffer = Factory::newBuffer(size, (BufferType)test->getEnum(ET_BUFFER), (ImplementationType)test->getEnum(ET_IMPLEMENTATION));
	Connection* connection = Factory::newConnection(buffer, size, (ImplementationType)test->getEnum(ET_IMPLEMENTATION));

	delete(connection);
	delete(buffer);

    checkAndPrintErrors("Connection", test);
}

void testLayer(Test* test)
{
    Layer *layer = new Layer(size, (BufferType)test->getEnum(ET_BUFFER), IDENTITY, (ImplementationType)test->getEnum(ET_IMPLEMENTATION));
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    delete (layer);

    checkAndPrintErrors("Layer", test);
}

//int a, b, c, d;
//unsigned miFuncioncita(Test* test){
//	test->printCurrentState();
//	return Random::positiveInteger(2);
//}

int main(int argc, char *argv[]) {

	Test test;
	Chronometer total;
	total.start();
	try {
//		test.withAll(ET_BUFFER);
//		test.withAll(ET_IMPLEMENTATION);
//		test.withAll(ET_CROSS_ALG);
//		test.withAll(ET_CROSS_LEVEL);
//		test.withAll(ET_MUTATION_ALG);
//		test.withAll(ET_FUNCTION);
//		test.addIterator(&a, 1, 2, 1);
//		test.addIterator(&b, 1, 2, 1);
//		test.addIterator(&c, 1, 2, 1);
//		test.addIterator(&d, 1, 2, 1);
//		test.test(miFuncioncita, "afdgfdgd");

		test.addIterator(&size, 100, 100, 100);
		test.withAll(ET_BUFFER);
		test.withAll(ET_IMPLEMENTATION);
		test.printParameters();

		test.simpleTest(testBuffer, "Buffer::memory");

		test.exclude(ET_BUFFER, 1, BYTE);
		test.printParameters();

		test.simpleTest(testConnection, "Connection::memory");
		test.simpleTest(testLayer, "Layer::memory");

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
