
#include <iostream>
#include <fstream>

using namespace std;

#include "test.h"
#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"


int main(int argc, char *argv[]) {

	Test test;
	Chronometer total;
	unsigned memoryLosses = 0;

	test.setMaxSize(500);
	test.setIncSize(100);
	test.printParameters();

	total.start();
	try {

		for (test.sizeToMin(); test.sizeIncrement(); ) {
			for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {
				for (test.implementationTypeToMin(); test.implementationTypeIncrement(); ) {

					Vector* vector = Factory::newVector(test.getSize(), test.getVectorType(), test.getImplementationType());
					delete(vector);

					if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
						cout << "Memory loss detected testing class Vector.\n" << endl;
						test.printCurrentState();
						mem_printTotalAllocated();
						mem_printTotalPointers();
						memoryLosses++;
					}
					if (test.getVectorType() != BYTE){

						vector = Factory::newVector(test.getSize(), test.getVectorType(), test.getImplementationType());
						Connection* connection = Factory::newConnection(vector, test.getSize(), test.getImplementationType());

						delete(connection);
						delete(vector);

						if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
							cout << "Memory loss detected testing class Connection.\n" << endl;
							test.printCurrentState();
							mem_printTotalAllocated();
							mem_printTotalPointers();
							memoryLosses++;
						}

						Layer* layer = new Layer(test.getSize(), test.getVectorType(), IDENTITY, test.getImplementationType());
						layer->addInput(layer->getOutput());
						layer->addInput(layer->getOutput());
						layer->addInput(layer->getOutput());

						delete(layer);

						if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
							cout << "Memory loss detected testing class Layer.\n" << endl;
							test.printCurrentState();
							mem_printTotalAllocated();
							mem_printTotalPointers();
							memoryLosses++;
						}
					}
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
