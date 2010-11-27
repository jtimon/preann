
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"


void printTestParams(ImplementationType implementationType, VectorType vectorType, unsigned size)
{
    switch (implementationType){
        case C: 		printf(" C        "); 	break;
        case SSE2: 		printf(" SSE2     ");	break;
        case CUDA: 		printf(" CUDA     ");	break;
        case CUDA2:		printf(" CUDA2    ");	break;
        case CUDA_INV:	printf(" CUDA_INV ");	break;
    }
    switch (vectorType){
        case FLOAT: printf(" FLOAT "); 	break;
        case BIT: 	printf(" BIT   ");	break;
        case SIGN: 	printf(" SIGN  ");	break;
        case BYTE:	printf(" BYTE  ");	break;
    }
    printf(" size = %d \n", size);
}

#define SIZE_MIN 1
#define SIZE_MAX 500
#define SIZE_INC 100

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();
	try {

		for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size += SIZE_INC) {
			for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
				VectorType vectorType = (VectorType)((vectType));
				for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {
					ImplementationType implementationType = (ImplementationType)((implType));

					printTestParams(implementationType, vectorType, size);
					printf("-----------Vector-----------\n");
					Vector* vector = Factory::newVector(size, vectorType, implementationType);

					mem_printTotalAllocated();
					mem_printTotalPointers();
					printf("------------------\n");
					delete(vector);

					if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
						std::string error = "Memory loss detected testing class Vector.\n";
						throw error;
					}
					if (vectorType != BYTE){
						printf("-----------Connection-----------\n");
						vector = Factory::newVector(size, vectorType, implementationType);
						Connection* connection = Factory::newConnection(vector, size, implementationType);

						mem_printTotalAllocated();
						mem_printTotalPointers();
						printf("------------------\n");
						delete(connection);
						delete(vector);

						if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
							std::string error = "Memory loss detected testing class Connection.\n";
							throw error;
						}

						printf("-----------Layer-----------\n");
						Layer* layer = new Layer(size, vectorType, IDENTITY, implementationType);
						layer->addInput(layer->getOutput());
						layer->addInput(layer->getOutput());
						layer->addInput(layer->getOutput());

						mem_printTotalAllocated();
						mem_printTotalPointers();
						printf("------------------\n");
						delete(layer);

						if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
							std::string error = "Memory loss detected testing class Layer.\n";
							throw error;
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

	mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
