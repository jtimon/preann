
#include <iostream>
#include <fstream>

using namespace std;

#include "test.h"
#include "population.h"
#include "chronometer.h"

unsigned memoryLosses = 0;

void checkAndPrintErrors(string testingClass, Test test)
{
    if(mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0){
        cout << "Memory loss detected testing class " << testingClass << ".\n" << endl;
        test.printCurrentState();
        mem_printTotalAllocated();
        mem_printTotalPointers();
        memoryLosses++;
    }
}

void testVector(Test test)
{
	Vector* vector = Factory::newVector(test.getSize(), test.getVectorType(), test.getImplementationType());
	delete(vector);

    checkAndPrintErrors("Vector", test);
}

void testConnection(Test test)
{
	Vector* vector = Factory::newVector(test.getSize(), test.getVectorType(), test.getImplementationType());
	Connection* connection = Factory::newConnection(vector, test.getSize(), test.getImplementationType());

	delete(connection);
	delete(vector);

    checkAndPrintErrors("Connection", test);
}

void testLayer(Test test)
{
    Layer *layer = new Layer(test.getSize(), test.getVectorType(), IDENTITY, test.getImplementationType());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    delete (layer);

    checkAndPrintErrors("Layer", test);
}

int main(int argc, char *argv[]) {

	Test test;
	Chronometer total;

	test.setMaxSize(500);
	test.setIncSize(100);
	test.printParameters();

	total.start();
	try {

		for (test.sizeToMin(); test.sizeIncrement(); ) {
			for (test.implementationTypeToMin(); test.implementationTypeIncrement(); ) {

				for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {
					testVector(test);
				}

				test.disableVectorType(BYTE);
				for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {
					testConnection(test);
					testLayer(test);
				}
			}
		}

		printf("Exit success.\n", 1);
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
	} catch (...) {
		printf("An error was thrown.\n", 1);
	}

	cout << "Total memory losses: " << memoryLosses << endl;
	mem_printListOfPointers();

	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
