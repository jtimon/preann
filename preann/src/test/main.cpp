#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"

#define PATH "/home/timon/test.nn"

//TODO TCC esto peta con BIT inputType = BIT maxSize = 163840, 16258024
//este to do estaba en CppLayer::randomweighs

float testNeuralNet(NeuralNet* nn, unsigned inputSize, BufferType bufferType,
		unsigned times) {

	Chronometer chrono;
	nn->addInputLayer(inputSize, bufferType);

	FILE* stream = fopen(PATH, "r+b");
	nn->load(stream);
	fclose(stream);

	chrono.start();
	for (unsigned i = 0; i < times; i++) {
		nn->calculateOutput();
	}
	chrono.stop();

	nn->getOutput(0)->print();

	delete (nn);
	//MemoryManagement.printTotalAllocated();
	//MemoryManagement.mem_printTotalPointers();
	return chrono.getSeconds();
}

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();
	try {

		Chronometer chrono;
		NeuralNet* nn;
		Population* population;

		FILE* stream;
		BufferType inputType;
		FunctionType functionType;
		unsigned rangeWeighs = 20;
		unsigned numlayers;
		unsigned maxSize;
		unsigned times;
		unsigned size;
		float seconds;
		unsigned type;

		for (type = 1; type < 2; type++) {

			switch (type) {
			case 0:
				cout << "version float" << endl;
				inputType = FLOAT;
				functionType = IDENTITY;
				maxSize = 7500;
				break;
			case 1:
				cout << "version bit" << endl;
				inputType = BIT;
				functionType = BINARY_STEP;
				maxSize = 1025;
				break;
			case 2:
				cout << "version sign" << endl;
				inputType = SIGN;
				functionType = BIPOLAR_STEP;
				maxSize = 512;
				break;
			}

			numlayers = 2;
			times = 1;
			for (size = maxSize; size <= maxSize; size += 32) {
				cout << "size: " << size << endl;
				nn = new NeuralNet();

				MemoryManagement.printTotalAllocated();
				MemoryManagement.printTotalPointers();
				//nn->createFeedForwardNet(size, inputType, numlayers, size, inputType, functionType);
				//TODO TCC petaba con 16 < numlayers <=30 y size 512 (en Desktop)
				nn->createFullyConnectedNet(size, inputType, numlayers, size, inputType,
						functionType);

				nn->randomWeighs(rangeWeighs);
				stream = fopen(PATH, "w+b");
				nn->save(stream);
				fclose(stream);
				delete (nn);
				MemoryManagement.printTotalAllocated();
				MemoryManagement.printTotalPointers();

				nn = new NeuralNet(C);
				seconds = testNeuralNet(nn, size, inputType, times);
				printf("C++ %f \n", seconds);

				nn = new NeuralNet(SSE2);
				seconds = testNeuralNet(nn, size, inputType, times);
				printf("XMM %f \n", seconds);

				//			for (unsigned algorithm = 0; algorithm < 1; algorithm++){
				//				//algorithm = 2;
				//				printf("CUDA [algorithm %d]  ", algorithm);
				//				CudaLayer::algorithm = algorithm;
				//				for (unsigned blockSize = 512; blockSize <=512; blockSize *= 2){
				//					Cuda_Threads_Per_Block = blockSize;
				//					nn = new NeuralNet(CUDA);
				//					seconds = testNeuralNet(nn, size, inputType, times);
				//					printf("(%d) %f ", blockSize, seconds);
				//				}
				//				printf("\n", 1);
				//			}

				printf("CUDA2 ", 1);
				for (unsigned blockSize = 512; blockSize <= 512; blockSize *= 2) {
					Cuda_Threads_Per_Block = blockSize;
					nn = new NeuralNet(CUDA2);
					seconds = testNeuralNet(nn, size, inputType, times);
					printf("(%d) %f ", blockSize, seconds);
				}
				printf("\n", 1);

			}
		}

		printf("Exit success.\n", 1);
		MemoryManagement.printTotalAllocated();
		MemoryManagement.printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
	} catch (...) {
		printf("An error was thrown.\n", 1);
	}
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
