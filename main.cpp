#include <iostream>
#include <fstream>

using namespace std;

#include "xmmNeuralNet.h"
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

	printTotalAllocated();
	printTotalPointers();
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
try{
	Chronometer total;
	total.start();

    ofstream cTime;
    ofstream xmmTime;
    ofstream cudaTime;

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

	for (type=0; type < 2; type++){
/*
		switch(type){
		case 0:
			cout<<"version float"<<endl;
			inputType = FLOAT;
			functionType = IDENTITY;
			maxSize = 2048;
		    cTime.open ("cFeedForwardFloatTime.dat");
		    xmmTime.open ("xmmFeedForwardFloatTime.dat");
		    cudaTime.open ("cudaFeedForwardFloatTime.dat");
			break;
		case 1:
			cout<<"version bit"<<endl;
			inputType = BIT;
			functionType = BINARY_STEP;
			maxSize = 4096;
		    cTime.open ("cFeedForwardBitTime.dat");
		    xmmTime.open ("xmmFeedForwardBitTime.dat");
		    cudaTime.open ("cudaFeedForwardBitTime.dat");
			break;
		case 2:
			cout<<"version sign"<<endl;
			inputType = SIGN;
			functionType = BIPOLAR_STEP;
			maxSize = 4096;http://ubunthttp://ubuntuguide.org/wiki/Ubuntu:Karmic#gftphttp://ubuntuguide.org/wiki/Ubuntu:Karmic#gftpihuguide.org/wiki/Ubuntu:Karmic#gftp
		    cTime.open ("cFeedForwardSignTime.dat");
		    xmmTime.open ("xmmFeedForwardSignTime.dat");
		    cudaTime.open ("cudaFeedForwardSignTime.dat");
			break;
		}

		numlayers = 3;
		times = 1;
		for(size=32; size <= maxSize; size += 32){
			cout<<"size: "<<size<<endl;
			nn = new NeuralNet();
			input = nn->newVector(size, inputType);
			nn->addInput(input);
			nn->createFeedForwardNet(numlayers, size, inputType, functionType);
			printTotalAllocated();
			printTotalPointers();
			nn->randomWeighs(rangeWeighs);
			stream = fopen(PATH, "w+b");
			nn->save(stream);
			fclose(stream);
			delete(nn);
			delete(input);
			printTotalAllocated();
			printTotalPointers();

			//C++
			cout<<"version C++"<<endl;
			nn = new NeuralNet();
			input = nn->newVector(size, inputType);
			seconds = testNeuralNet(nn, input, times);
			cTime<<size<<"  "<<seconds<<endl;
			printTotalAllocated();
			printTotalPointers();
			//XMM
			cout<<"version XMM"<<endl;
			nn = new XmmNeuralNet();
			input = nn->newVector(size, inputType);
			seconds = testNeuralNet(nn, input, times);
			xmmTime<<size<<"  "<<seconds<<endl;
			printTotalAllocated();
			printTotalPointers();
			//CUDA
			cout<<"version CUDA"<<endl;
			nn = new CudaNeuralNet();
			input = nn->newVector(size, inputType);
			seconds = testNeuralNet(nn, input, times);
			cudaTime<<size<<"  "<<seconds<<endl;
			printTotalAllocated();
			printTotalPointers();
		}
		cTime.close();
		xmmTime.close();
		cudaTime.close();*/

		switch(type){
		case 0:
			inputType = FLOAT;
			functionType = IDENTITY;
			maxSize = 1024;
		    cTime.open ("cFullyConnectedFloatTime.dat");
		    xmmTime.open ("xmmFullyConnectedFloatTime.dat");
		    cudaTime.open ("cudaFullyConnectedFloatTime.dat");
			break;
		case 1:
			inputType = BIT;
			functionType = BINARY_STEP;
			maxSize = 2048;
		    cTime.open ("cFullyConnectedBitTime.dat");
		    xmmTime.open ("xmmFullyConnectedBitTime.dat");
		    cudaTime.open ("cudaFullyConnectedBitTime.dat");
			break;
		case 2:
			inputType = SIGN;
			functionType = BIPOLAR_STEP;
			maxSize = 2048;
		    cTime.open ("cFullyConnectedSignTime.dat");
		    xmmTime.open ("xmmFullyConnectedSignTime.dat");
		    cudaTime.open ("cudaFullyConnectedSignTime.dat");
			break;
		}
		times = 1;
		for(numlayers=2; numlayers < 100; numlayers++){
			for(unsigned size=32; size <= maxSize; size += 32){
				nn = new NeuralNet();
				input = nn->newVector(size, inputType);
				nn->addInput(input);
				nn->createFullyConnectedNet(numlayers, size, inputType, functionType);
				nn->randomWeighs(rangeWeighs);
				stream = fopen(PATH, "w+b");
				nn->save(stream);
				fclose(stream);
				printTotalAllocated();
				printTotalPointers();
				delete(nn);
				delete(input);
				printTotalAllocated();
				printTotalPointers();
				//C++
				cout<<"version C++"<<endl;
				nn = new NeuralNet();
				input = nn->newVector(size, inputType);
				seconds = testNeuralNet(nn, input, times);
				cTime<<numlayers<<"  "<<size<<"  "<<seconds<<endl;
				printTotalAllocated();
				printTotalPointers();
				//XMM
				cout<<"version XMM"<<endl;
				nn = new XmmNeuralNet();
				input = nn->newVector(size, inputType);
				seconds = testNeuralNet(nn, input, times);
				xmmTime<<numlayers<<"  "<<size<<"  "<<seconds<<endl;
				printTotalAllocated();
				printTotalPointers();
				//CUDA
				cout<<"version CUDA"<<endl;
				nn = new CudaNeuralNet();
				input = nn->newVector(size, inputType);
				seconds = testNeuralNet(nn, input, times);
				cudaTime<<numlayers<<"  "<<size<<"  "<<seconds<<endl;
				printTotalAllocated();
				printTotalPointers();
			}
		}
		cTime.close();
		xmmTime.close();
		cudaTime.close();
	}

	total.stop();
	cout<<"Total time spent: "<<chrono.getSeconds()<<endl;
	cout<<"Exit success"<<endl;

} catch(string error){
	cout<<"Error: "<<error<<endl;
} catch (...){
	cout<<"An error was thrown."<<endl;
}
	printTotalAllocated();
	printTotalPointers();
	return EXIT_SUCCESS;
}
