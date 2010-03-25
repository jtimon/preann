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

	nn->getOutput(0)->print();
	//printTotalAllocated();
	//printTotalPointers();
	delete(nn);

	return chrono.getSeconds();
}

float testLayer(Layer* layer, Vector* input, unsigned times){

	Chronometer chrono;
	FILE* stream = fopen("test.lay", "r+b");
	layer->load(stream);
	fclose(stream);
	layer->addInput(input);
	printTotalAllocated();
	printTotalPointers();
	chrono.start();
	for (unsigned i=0; i < times; i++){
		layer->calculateOutput();
	}
	//layer->getOutput()->showVector();
	chrono.stop();
	delete(layer);
	delete(input);

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

	//for (type=1; type < 3; type++){

		type = 1;
		switch(type){
		case 0:
			cout<<"version float"<<endl;
			inputType = FLOAT;
			functionType = IDENTITY;
			maxSize = 512;
			break;
		case 1:
			cout<<"version bit"<<endl;
			inputType = BIT;
			functionType = BINARY_STEP;
			maxSize = 512;
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

			//C++
			//cout<<"version C++"<<endl;
			nn = new NeuralNet(C);
			float c_seconds = testNeuralNet(nn, size, inputType, times);
			printf("c %f \n", c_seconds);
			//XMM
			//cout<<"version XMM"<<endl;
			nn = new NeuralNet(SSE2);
			float xmm_seconds = testNeuralNet(nn, size, inputType, times);
			printf("xmm %f \n", xmm_seconds);
			//CUDA
			//cout<<"version CUDA"<<endl;
			for (unsigned version = 2; version < 3; version++){
				printf("cuda version %d  ", version);
				CudaLayer::version = version;
				for (unsigned blockSize = 512; blockSize <=512; blockSize *= 2){
					CudaLayer::block_size = blockSize;
					nn = new CudaNeuralNet();
					float cuda_seconds = testNeuralNet(nn, size, inputType, times);
					printf("(%d) %f ", blockSize, cuda_seconds);
				}
				printf("\n", 1);
			}
		}
	//}

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
