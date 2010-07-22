
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
#define MAX_INPUTS 3
#define INITIAL_WEIGHS_RANGE 20
#define NUM_LOOPS 3

void assertEquals(Interface* expected, Interface* actual)
{
	if (expected->getVectorType() != actual->getVectorType()){
		throw "The interfaces are not even of the same type!";
	}
	if (expected->getSize() != actual->getSize()){
		throw "The interfaces are not even of the same size!";
	}
	for (unsigned i=0; i < expected->getSize(); i++) {
		if ((int)expected->getElement(i) != (int)actual->getElement(i)){
			printf("expected:\n");
			expected->print();
			printf("actual:\n");
			actual->print();
			char buffer[100];
			sprintf(buffer, "The interfaces are not equal at the position %d.", i);
			string error = buffer;
			throw error;
		}
	}
}

#define PATH "/home/timon/layer.lay"

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();
	try {
		for (unsigned vectType = 2; vectType < VECTOR_TYPE_DIM; vectType++) {
			for (unsigned size = 1; size < MAX_SIZE; size++){

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

				Layer* layer = Factory::newLayer(C);
				layer->init(size, vectorType, IDENTITY);

				for (unsigned numInputs = 1; numInputs < MAX_INPUTS; numInputs++){
					layer->addInput(layer->getOutput());
				}

				layer->randomWeighs(INITIAL_WEIGHS_RANGE);
				FILE* stream = fopen(PATH, "w+b");
				layer->saveWeighs(stream);

				for(unsigned numLoops = 0; numLoops < NUM_LOOPS; numLoops++){
					layer->calculateOutput();
				}

				Interface* expected = layer->getOutput()->createInterface();

				fclose(stream);
				delete(layer);

				for (unsigned implType = 0; implType < IMPLEMENTATION_TYPE_DIM; implType++) {

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

					for (unsigned numInputs = 1; numInputs < MAX_INPUTS; numInputs++){
						layer->addInput(layer->getOutput());
					}

					FILE* stream = fopen(PATH, "r+b");
					layer->loadWeighs(stream);
					fclose(stream);

					for(unsigned numLoops = 0; numLoops < NUM_LOOPS; numLoops++){
						layer->calculateOutput();
					}

					Interface* actual = layer->getOutput()->createInterface();

					assertEquals(expected, actual);

					delete (actual);
					delete(layer);
				}
				delete (expected);
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
