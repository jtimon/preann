#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "cudaLayer2.h"

#define PATH "/home/timon/test.nn"
//TODO pensar una forma de poner un error razonable
#define ALLOWED_ERROR (0.01)

void assertEquals(Interface* expected, Interface* actual)
{
	if (expected->getVectorType() != actual->getVectorType()){
		throw "The interfaces are not even of the same type!";
	}
	if (expected->getVectorType() == FLOAT){
		float allowedError = ALLOWED_ERROR * expected->getSize();
		printf("allowedError %f\n", allowedError);
		for (unsigned i=0; i < expected->getSize(); i++) {

			float difference = (expected->getElement(i) / 100) * (expected->getElement(i) - actual->getElement(i));

			if ((difference > 0 && difference > allowedError) ||
					(difference < 0 && difference < - allowedError)){
				printf("expected:\n");
				expected->print();
				printf("actual:\n");
				actual->print();
				char buffer[100];
				sprintf(buffer, "The interfaces are not equal at the position %d.", i);
				std::string error = buffer;
				throw error;
			}
		}
	} else {
		for (unsigned i=0; i < expected->getSize(); i++) {
			if (expected->getElement(i) != actual->getElement(i)){
				printf("expected:\n");
				expected->print();
				printf("actual:\n");
				actual->print();
				char buffer[100];
				sprintf(buffer, "The interfaces are not equal at the position %d.", i);
				std::string error = buffer;
				throw error;
			}
		}
	}
}

Interface* testNeuralNet(NeuralNet* nn, unsigned inputSize, VectorType vectorType, unsigned times, Chronometer& chrono) {

	nn->createInput(inputSize, vectorType);

	FILE* stream = fopen(PATH, "r+b");
	nn->load(stream);
	fclose(stream);

	chrono.start();
	for (unsigned i = 0; i < times; i++) {
		nn->calculateOutput();
	}
	chrono.stop();

	Interface* result = new Interface(nn->getOutput(0));
	delete (nn);
//	mem_printTotalAllocated();
//	mem_printTotalPointers();
	return result;
}

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();
	try {

		Chronometer chrono;
		NeuralNet* nn;
		Population* population;

		FILE* stream;
		VectorType inputType;
		FunctionType functionType;
		unsigned minSize, maxSize, sizeIncrease;

		unsigned rangeWeighs = 20;
		unsigned numlayers = 2;
		unsigned times = 1;

		for (unsigned type = 0; type < 3; type++) {

			switch (type) {
			case 0:
				cout << "version float" << endl;
				inputType = FLOAT;
				functionType = IDENTITY;
				minSize = 32;
				maxSize = 2000;
				sizeIncrease = 500;
				break;
			case 1:
				cout << "version bit" << endl;
				inputType = BIT;
				functionType = BINARY_STEP;
				minSize = 32;
				maxSize = 2000;
				sizeIncrease = 500;
				break;
			case 2:
				cout << "version sign" << endl;
				inputType = SIGN;
				functionType = BIPOLAR_STEP;
				minSize = 32;
				maxSize = 2000;
				sizeIncrease = 500;
				break;
			}

			for (unsigned size = minSize; size <= maxSize; size += sizeIncrease) {
				cout << "size: " << size << endl;
				nn = new NeuralNet();

				nn->createInput(size, inputType);
				//nn->createFeedForwardNet(numlayers, size, inputType, functionType);
				//TODO petaba con 16 < numlayers <=30 y size 512 (en Desktop)
				nn->createFullyConnectedNet(numlayers, size, inputType, functionType);

				nn->randomWeighs(rangeWeighs);
				stream = fopen(PATH, "w+b");
				nn->save(stream);
				fclose(stream);
				delete (nn);

				nn = new NeuralNet(C);
				Interface* cppResult = testNeuralNet(nn, size, inputType, times, chrono);
				printf("C++ %f \n", chrono.getSeconds());

				nn = new NeuralNet(SSE2);
				Interface* xmmResult = testNeuralNet(nn, size, inputType, times, chrono);
				printf("XMM %f \n", chrono.getSeconds());
				assertEquals(cppResult, xmmResult);
				delete(xmmResult);

				for (unsigned algorithm = 0; algorithm < 1; algorithm++){
					//algorithm = 2;
					printf("CUDA [algorithm %d]  ", algorithm);
					CudaLayer::algorithm = algorithm;
					for (unsigned blockSize = 512; blockSize <=512; blockSize *= 2){
						Cuda_Threads_Per_Block = blockSize;
						nn = new NeuralNet(CUDA);
						Interface* cudaResult = testNeuralNet(nn, size, inputType, times, chrono);
						printf("(%d) %f ", blockSize, chrono.getSeconds());
						assertEquals(cppResult, cudaResult);
						delete(cudaResult);
					}
					printf("\n", 1);
				}

				printf("CUDA2 ", 1);
				for (unsigned blockSize = 512; blockSize <= 512; blockSize *= 2) {
					Cuda_Threads_Per_Block = blockSize;
					nn = new NeuralNet(CUDA2);
					Interface* cudaResult = testNeuralNet(nn, size, inputType, times, chrono);
					printf("(%d) %f ", blockSize, chrono.getSeconds());
					assertEquals(cppResult, cudaResult);
					delete(cudaResult);
				}
				printf("\n", 1);
				delete(cppResult);
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
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
