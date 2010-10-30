
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "cudaVector.h"

#define INITIAL_WEIGHS_RANGE 20

void printTestParams(ImplementationType implementationType, VectorType vectorType, unsigned size, unsigned numInputs)
{
    switch (implementationType){
        case C: 	printf(" C     "); 	break;
        case SSE2: 	printf(" SSE2  ");	break;
        case CUDA: 	printf(" CUDA  ");	break;
        case CUDA2:	printf(" CUDA2 ");	break;
    }
    switch (vectorType){
        case FLOAT: printf(" FLOAT "); 	break;
        case BIT: 	printf(" BIT   ");	break;
        case SIGN: 	printf(" SIGN  ");	break;
        case BYTE:	printf(" BYTE  ");	break;
    }
    printf(" size = %d numInputs %d weighsRange %d \n", size, numInputs, INITIAL_WEIGHS_RANGE);
}

unsigned char areEqual(float expected, float actual, VectorType vectorType)
{
	if (vectorType == FLOAT){
		return (expected - 1 < actual
			 && expected + 1 > actual);
	} else {
		return expected == actual;
	}
}



#define IMPLEMENTATION_TYPE_DIM 2
#define VECTOR_TYPE_DIM 3
#define SIZE_MAX 5
#define SIZE_INC 1
#define PATH "/home/timon/layer.lay"

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();

	try {
		for (unsigned vectType = 0; vectType < VECTOR_TYPE_DIM; vectType++) {
			for (unsigned size = 1; size < SIZE_MAX; size += SIZE_INC) {
				Interface* control = new Interface(size, (VectorType)vectType);
				control->random(INITIAL_WEIGHS_RANGE);

				Vector* vector = Factory::newVector(size, (VectorType)vectType, C);
				vector->random(INITIAL_WEIGHS_RANGE);

				Vector* another = vector->clone();
			}
		}

		printf("Exit success.\n");
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
//	} catch (...) {
//		printf("An error was thrown.\n", 1);
	}

	//mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
