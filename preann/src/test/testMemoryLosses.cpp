
#include <iostream>
#include <fstream>

using namespace std;

#include "test.h"
#include "population.h"
#include "chronometer.h"

unsigned memoryLosses = 0;

void checkAndPrintErrors(string testingClass, Test test)
{
    if(MemoryManagement::getPtrCounter() > 0 || MemoryManagement::getTotalAllocated() > 0){
        cout << "Memory loss detected testing class " << testingClass << ".\n" << endl;
        test.printCurrentState();
        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
        memoryLosses++;
    }
}

void testBuffer(Test test)
{
	Buffer* buffer = Factory::newBuffer(test.getSize(), test.getBufferType(), test.getImplementationType());
	delete(buffer);

    checkAndPrintErrors("Buffer", test);
}

void testConnection(Test test)
{
	Buffer* buffer = Factory::newBuffer(test.getSize(), test.getBufferType(), test.getImplementationType());
	Connection* connection = Factory::newConnection(buffer, test.getSize(), test.getImplementationType());

	delete(connection);
	delete(buffer);

    checkAndPrintErrors("Connection", test);
}

void testLayer(Test test)
{
    Layer *layer = new Layer(test.getSize(), test.getBufferType(), IDENTITY, test.getImplementationType());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    delete (layer);

    checkAndPrintErrors("Layer", test);
}

int main(int argc, char *argv[]) {

	Test test;
	Chronometer total;

	test.setMaxSize(100);
	test.setIncSize(100);
	test.printParameters();

	total.start();
	try {

		//TODO usar Test para esto (sin el bucle este aqui fuera) o eliminar el fichero
		for (test.sizeToMin(); test.hasNextSize(); test.sizeIncrement()) {
			for (test.implementationTypeToMin(); test.hasNextImplementationType(); test.implementationTypeIncrement()) {

				for (test.bufferTypeToMin(); test.hasNextBufferType(); test.bufferTypeIncrement() ) {
					testBuffer(test);
				}

				test.disableBufferType(BYTE);
				for (test.bufferTypeToMin(); test.hasNextBufferType(); test.bufferTypeIncrement() ) {
					testConnection(test);
					testLayer(test);
				}
			}
		}

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
