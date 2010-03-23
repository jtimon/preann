#include <iostream>
#include <fstream>

using namespace std;

#include "cudaNeuralNet.h"
#include "population.h"
#include "chronometer.h"

#define PATH "/home/timon/test.nn"

float testNeuralNet(NeuralNet* nn, Vector* input, unsigned times){

	Chronometer chrono;
	nn->addInput(input);
	FILE* stream = fopen(PATH, "r+b");
	nn->load(stream);
	fclose(stream);

	chrono.start();
	for (unsigned i=0; i < times; i++){
		nn->calculateOutput();
	}
	chrono.stop();

	//nn->getOutput(0)->showVector();
	//printTotalAllocated();
	//printTotalPointers();
	delete(nn);
	delete(input);

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
	layer->getOutput()->showVector();
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
	Vector* input;
	Vector* output;
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

	//for (type=1; type < 2; type++){

		type = 1;
		switch(type){
		case 0:
			cout<<"version float"<<endl;
			inputType = FLOAT;
			functionType = IDENTITY;
			maxSize = 4096;
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

		numlayers = 2;
		times = 1;
		for(size=maxSize; size <= maxSize; size += 32){
			cout<<"size: "<<size<<endl;
			nn = new NeuralNet();
			input = nn->newVector(size, inputType);
			nn->addInput(input);

			nn->createFeedForwardNet(numlayers, size, inputType, functionType);
			//TODO peta con 16 < numlayers <=30 y size 512
			//nn->createFullyConnectedNet(numlayers, size, inputType, functionType);

			//printTotalAllocated();
			//printTotalPointers();
			nn->randomWeighs(rangeWeighs);
			stream = fopen(PATH, "w+b");
			nn->save(stream);
			fclose(stream);
			delete(nn);
			delete(input);
			printTotalAllocated();
			printTotalPointers();

			//C++
			//cout<<"version C++"<<endl;
			nn = new NeuralNet();
			input = nn->newVector(size, inputType);
			float c_seconds = testNeuralNet(nn, input, times);
			printf("c %f \n", c_seconds);
			printTotalAllocated();
			printTotalPointers();
			//XMM
			//cout<<"version XMM"<<endl;
			nn = new NeuralNet(SSE2);
			input = nn->newVector(size, inputType);
			float xmm_seconds = testNeuralNet(nn, input, times);
			printf("xmm %f \n", xmm_seconds);
			printTotalAllocated();
			printTotalPointers();
			//CUDA
			//cout<<"version CUDA"<<endl;
			for (unsigned version = 2; version < 3; version++){
				printf("cuda version %d  ", version);
				CudaLayer::version = version;
				for (unsigned blockSize = 512; blockSize <=512; blockSize *= 2){
					CudaLayer::block_size = blockSize;
					nn = new CudaNeuralNet();
					input = nn->newVector(size, inputType);
					float cuda_seconds = testNeuralNet(nn, input, times);
					printf("(%d) %f ", blockSize, cuda_seconds);
				}
				printf("\n", 1);
			}
			printTotalAllocated();
			printTotalPointers();
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
