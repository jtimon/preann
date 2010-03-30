#include <iostream>
#include <fstream>

using namespace std;

#include "cudaNeuralNet.h"
#include "population.h"
#include "chronometer.h"

#define PATH "/home/timon/test.nn"

float testNeuralNet(NeuralNet* nn, unsigned inputSize, VectorType vectorType, unsigned times){

	Chronometer chrono;
	nn->createInput(inputSize, vectorType);
	FILE* stream = fopen(PATH, "r+b");
	nn->load(stream);
	fclose(stream);

	chrono.start();
	for (unsigned i=0; i < times; i++){
		nn->calculateOutput();
	}
	chrono.stop();

	//nn->getOutput(0)->print();
	//printTotalAllocated();
	//printTotalPointers();
	delete(nn);

	return chrono.getSeconds();
}

unsigned numlayers;
unsigned type;
unsigned size;

int main ( int argc, char *argv[] )
{
	Chronometer total;
	total.start();
try{
	Chronometer chrono;
	NeuralNet* nn;
	Layer* layer;
	FILE* stream;
	VectorType inputType;
	FunctionType functionType;
	unsigned rangeWeighs = 20;
	unsigned maxSize;
	unsigned times;
	float seconds;
	FILE* ftimes = fopen("/home/timon/times.log", "w");

	for (type=0; type < 2; type++){

		//type = 1;
		switch(type){
		case 0:
			cout<<"version float"<<endl;
			inputType = FLOAT;
			functionType = IDENTITY;
			maxSize = 2048;
			break;
		case 1:
			cout<<"version bit"<<endl;
			inputType = BIT;
			functionType = BINARY_STEP;
			maxSize = 4096;
			break;
		case 2:
			cout<<"version sign"<<endl;
			inputType = SIGN;
			functionType = BIPOLAR_STEP;
			maxSize = 4096;
			break;
		}

		numlayers = 3;
		times = 1;
		for(size=maxSize; size <= maxSize; size += 32){
			cout<<"size: "<<size<<endl;
			nn = new NeuralNet();

			nn->createInput(size, inputType);
			//nn->createFeedForwardNet(numlayers, size, inputType, functionType);
			//TODO petaba con 16 < numlayers <=30 y size 512 (en Desktop)
			nn->createFullyConnectedNet(numlayers, size, inputType, functionType);

			//printTotalAllocated();
			//printTotalPointers();
			nn->randomWeighs(rangeWeighs);
			stream = fopen(PATH, "w+b");
			nn->save(stream);
			fclose(stream);
			delete(nn);
			printTotalAllocated();
			printTotalPointers();

			float secods;
			//C++
			//cout<<"version C++"<<endl;
			nn = new NeuralNet(C);
			secods = testNeuralNet(nn, size, inputType, times);
			printf("C++ %f \n", secods);
			//XMM
			//cout<<"version XMM"<<endl;
			nn = new NeuralNet(SSE2);
			secods = testNeuralNet(nn, size, inputType, times);
			printf("XMM %f \n", secods);
			//CUDA
			//cout<<"version CUDA"<<endl;
			for (unsigned version = 0; version < 3; version++){
				printf("cuda OLD [version %d]  ", version);
				CudaLayer::version = version;
				for (unsigned blockSize = 8; blockSize <=512; blockSize *= 2){
					CudaLayer::block_size = blockSize;
					nn = new CudaNeuralNet();
					secods = testNeuralNet(nn, size, inputType, times);
					printf("(%d) %f ", blockSize, secods);
				}
				printf("\n", 1);
			}
			for (unsigned algorithm = 0; algorithm < 2; algorithm++){
				printf("CUDA [algorithm %d]  ", algorithm);
				CudaLayer2::algorithm = algorithm;
				for (unsigned blockSize = 8; blockSize <=512; blockSize *= 2){
					CudaLayer2::blockSize = blockSize;
					nn = new NeuralNet(CUDA2);
					secods = testNeuralNet(nn, size, inputType, times);
					printf("(%d) %f ", blockSize, secods);
				}
				printf("\n", 1);
			}
		}
	}

	fclose(ftimes);
	cout<<"Exit success"<<endl;

} catch(string error){
	cout<<"Error: "<<error<<endl;
} catch (...){
	cout<<"An error was thrown."<<endl;
}
	total.stop();
	cout<<"Total time spent: "<<total.getSeconds()<<endl;
	printTotalAllocated();
	printTotalPointers();
	return EXIT_SUCCESS;
}
