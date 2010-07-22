
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "cudaLayer2.h"

#define VECTOR_TYPE_DIM 3
#define IMPLEMENTATION_TYPE_DIM 4
#define MAX_SIZE 200

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();
	try {

		//test Vectors
		for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
			for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {
				for (unsigned size = 1; size < MAX_SIZE; size++){

					printf(" Vector ");
					VectorType vectorType;
					switch (vectType) {
					case 0:
						vectorType = FLOAT;
						printf(" FLOAT ");
						break;
					case 1:
						vectorType = BIT;
						printf(" BIT ");
						break;
					case 2:
						vectorType = SIGN;
						printf(" SIGN ");
						break;
					}

					ImplementationType implementationType;
					switch (implType) {
					case 0:
						implementationType = C;
						printf(" C ");
						break;
					case 1:
						implementationType = SSE2;
						printf(" SSE2 ");
						break;
					case 2:
						implementationType = CUDA;
						printf(" CUDA ");
						break;
					case 3:
						implementationType = CUDA2;
						printf(" CUDA2 ");
						break;
					}

					printf(" size = %d \n", size);
					Vector* vector = Factory::newVector(size, vectorType, implementationType);

					mem_printTotalAllocated();
					mem_printTotalPointers();
					printf("------------------\n");
					delete(vector);

					if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
						string error = "Memory loss detected testing class Vector.\n";
						throw error;
					}
				}
			}
		}

		//test Layers
		for (unsigned vectType = 0; vectType < 3; vectType++) {
			for (unsigned implType = 0; implType < 4; implType++) {
				for (unsigned size = 1; size < 200; size++){

					printf(" Layer ");
					VectorType vectorType;
					switch (vectType) {
					case 0:
						vectorType = FLOAT;
						printf(" FLOAT ");
						break;
					case 1:
						vectorType = BIT;
						printf(" BIT ");
						break;
					case 2:
						vectorType = SIGN;
						printf(" SIGN ");
						break;
					}

					ImplementationType implementationType;
					switch (implType) {
					case 0:
						implementationType = C;
						printf(" C ");
						break;
					case 1:
						implementationType = SSE2;
						printf(" SSE2 ");
						break;
					case 2:
						implementationType = CUDA;
						printf(" CUDA ");
						break;
					case 3:
						implementationType = CUDA2;
						printf(" CUDA2 ");
						break;
					}
					printf(" size = %d \n", size);

					Layer* layer = Factory::newLayer(implementationType);
					layer->init(size, vectorType, IDENTITY);
					layer->addInput(layer->getOutput());
					layer->addInput(layer->getOutput());
					layer->addInput(layer->getOutput());

					mem_printTotalAllocated();
					mem_printTotalPointers();

					delete(layer);

					printf("-- after deleting --\n");
					mem_printTotalAllocated();
					mem_printTotalPointers();
					printf("-- -------------- --\n");

					if (mem_getPtrCounter() > 0 || mem_getTotalAllocated() > 0 ){
						string error = "Memory loss detected testing class Layer.\n";
						throw error;
					}
				}
			}
		}


		printf("Exit success.\n", 1);
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (string error) {
		cout << "Error: " << error << endl;
	} catch (...) {
		printf("An error was thrown.\n", 1);
	}

	mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
