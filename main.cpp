
#include <iostream>
#include <fstream>

using namespace std;

#include "xmmNeuralNet.h"
#include "cudaNeuralNet.h"
#include "chronometer.h"

float testNeuralNet(NeuralNet* nn, Vector* input, unsigned times){

	Chronometer chrono;
	nn->addInput(input);
	FILE* stream = fopen("test.nn", "r+b");
	nn->load(stream);
	fclose(stream);
	chrono.start();
	for (unsigned i=0; i < times; i++){
		nn->calculateOutput();
	}
	chrono.stop();
	delete(nn);
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
	FILE* stream;
	VectorType inputType;
	FunctionType functionType;
	unsigned rangeWeighs = 20;
	unsigned maxSize;
	unsigned times;
	float seconds;

	for (type=0; type < 3; type++){

		switch(type){
		case 0:
			inputType = FLOAT;
			functionType = IDENTITY;
			maxSize = 2048;
		    cTime.open ("cFeedForwardFloatTime.dat");
		    xmmTime.open ("xmmFeedForwardFloatTime.dat");
		    cudaTime.open ("cudaFeedForwardFloatTime.dat");
			break;
		case 1:
			inputType = BIT;
			functionType = BINARY_STEP;
			maxSize = 4096;
		    cTime.open ("cFeedForwardBitTime.dat");
		    xmmTime.open ("xmmFeedForwardBitTime.dat");
		    cudaTime.open ("cudaFeedForwardBitTime.dat");
			break;
		case 2:
			inputType = SIGN;
			functionType = BIPOLAR_STEP;
			maxSize = 4096;
		    cTime.open ("cFeedForwardSignTime.dat");
		    xmmTime.open ("xmmFeedForwardSignTime.dat");
		    cudaTime.open ("cudaFeedForwardSignTime.dat");
			break;
		}

		numlayers = 3;
		times = 50;
		for(size=32; size <= maxSize; size += 32){
			nn = new NeuralNet();
			input = nn->newVector(size, inputType);
			nn->addInput(input);
			nn->createFeedForwardNet(numlayers, size, inputType, functionType);
			nn->randomWeighs(rangeWeighs);
			stream = fopen("test.nn", "w+b");
			nn->save(stream);
			fclose(stream);
			delete(nn);
			delete(input);
			//C++
			nn = new NeuralNet();
			input = nn->newVector(size, inputType);
			seconds = testNeuralNet(nn, input, times);
			cTime<<size<<"  "<<seconds<<endl;
			//XMM
			nn = new XmmNeuralNet();
			input = nn->newVector(size, inputType);
			seconds = testNeuralNet(nn, input, times);
			xmmTime<<size<<"  "<<seconds<<endl;
			//CUDA
			nn = new CudaNeuralNet();
			input = nn->newVector(size, inputType);
			seconds = testNeuralNet(nn, input, times);
			cudaTime<<size<<"  "<<seconds<<endl;
		}
		cTime.close();
		xmmTime.close();
		cudaTime.close();

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
		times = 10;
		for(numlayers=2; numlayers < 100; numlayers++){
			for(unsigned size=32; size <= maxSize; size += 32){
				//cout<<endl<<"type: "<<inputType<<" layers: "<<numlayers<<" size: "<<size<<endl<<endl;
				nn = new NeuralNet();
				input = nn->newVector(size, inputType);
				nn->addInput(input);
				nn->createFullyConnectedNet(numlayers, size, inputType, functionType);
				nn->randomWeighs(rangeWeighs);
				stream = fopen("test.nn", "w+b");
				nn->save(stream);
				fclose(stream);
				delete(nn);
				delete(input);
				//C++
				nn = new NeuralNet();
				input = nn->newVector(size, inputType);
				seconds = testNeuralNet(nn, input, times);
				cTime<<numlayers<<"  "<<size<<"  "<<seconds<<endl;
				//XMM
				nn = new XmmNeuralNet();
				input = nn->newVector(size, inputType);
				seconds = testNeuralNet(nn, input, times);
				xmmTime<<numlayers<<"  "<<size<<"  "<<seconds<<endl;
				//CUDA
				nn = new CudaNeuralNet();
				input = nn->newVector(size, inputType);
				seconds = testNeuralNet(nn, input, times);
				cudaTime<<numlayers<<"  "<<size<<"  "<<seconds<<endl;
			}
		}
		cTime.close();
		xmmTime.close();
		cudaTime.close();
	}

	total.stop();
	cout<<"Total time spent: "<<chrono.getSeconds()<<endl;
	cout<<"Exit success"<<endl;

} catch (unsigned number){
	cout<<"An error ocurred related with this unsigned: "<<number<<endl;
} catch (...){
	cout<<"An error ocurred."<<endl;
}
	return EXIT_SUCCESS;
}
