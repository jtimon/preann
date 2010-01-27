
#include "xmmNeuralNet.h"
#include "cudaNeuralNet.h"
#include "chronometer.h"

int main ( int argc, char *argv[] )
{
	Chronometer chrono;
	Vector* input;
	Vector* output;
	NeuralNet* nn;
	FILE* stream;

	unsigned inputSize = 1000;
	VectorType inputType = BIT;
	FunctionType functionType = BINARY_STEP;

	nn = new NeuralNet();
	input = nn->newVector(inputSize, inputType);
	nn->addInput(input);
	nn->addInput(input);
	nn->addInput(input);
	nn->createFeedForwardNet(3, 500, inputType, functionType);

	nn->randomWeighs(20);

	stream = fopen("prueba.nn", "w+b");
	nn->save(stream);
	fclose(stream);

	delete(nn);
	delete(input);

	nn = new NeuralNet();

	input = nn->newVector(inputSize, inputType);
	nn->addInput(input);
	nn->addInput(input);
	nn->addInput(input);
	stream = fopen("prueba.nn", "r+b");
	nn->load(stream);
	output = nn->getOutput(0);
	cout<<"Input:"<<endl;
	input->showVector();

	chrono.start();
	for (unsigned i=0; i<50; i++){
		nn->calculateOutput();
	}
	chrono.stop();
	cout<<"tiempo utilizado: "<<chrono.getSeconds()<<endl;
	cout<<"Output:"<<endl;
	output->showVector();

	delete(nn);
	delete(input);

	nn = new XmmNeuralNet();

	input = nn->newVector(inputSize, inputType);
	nn->addInput(input);
	nn->addInput(input);
	nn->addInput(input);
	stream = fopen("prueba.nn", "r+b");
	nn->load(stream);
	output = nn->getOutput(0);
	cout<<"Input:"<<endl;
	input->showVector();

	chrono.start();
	for (unsigned i=0; i<50; i++){
		nn->calculateOutput();
	}
	chrono.stop();
	cout<<"tiempo utilizado: "<<chrono.getSeconds()<<endl;
	cout<<"Output:"<<endl;
	output->showVector();

	delete(nn);
	delete(input);

	nn = new CudaNeuralNet();

	input = nn->newVector(inputSize, inputType);
	nn->addInput(input);
	nn->addInput(input);
	nn->addInput(input);
	stream = fopen("prueba.nn", "r+b");
	nn->load(stream);
	output = nn->getOutput(0);

	cout<<"Input:"<<endl;
	input->showVector();

	((CudaNeuralNet*)nn)->hostToDevice();
	chrono.start();
	for (unsigned i=0; i<50; i++){
		nn->calculateOutput();
	}
	chrono.stop();
	cout<<"tiempo utilizado: "<<chrono.getSeconds()<<endl;
	cout<<"Output:"<<endl;
	output->showVector();

	((CudaNeuralNet*)nn)->freeDevice();
	delete(nn);
	delete(input);

/*
	delete (nn);

	nn = new NeuralNet();
	nn->addInput(input);
	nn->createFullyConnectedNet(6, 256, FLOAT);


	nn->randomWeighs(20);
	output = nn->getOutput(0);

	nn->calculateOutput();

	cout<<"Output:"<<endl;
	output->showVector();

	delete(nn);
	delete (input);*/
	cout<<"Finalizando con éxito"<<endl;
	return EXIT_SUCCESS;
}
