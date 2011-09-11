
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "test.h"

int size;
unsigned outputSize = 100;
float initialWeighsRange = 20;

#define PATH "/home/timon/layer.lay"
#define NUM_INPUTS 3

Layer* createAndLoadLayer(ImplementationType implementationType, Buffer** inputBuffers)
{
    FILE* stream = fopen(PATH, "r+b");
    Layer* layer = new Layer(stream, implementationType);

    for (unsigned i = 0; i < NUM_INPUTS; i++){
		layer->addInput(inputBuffers[i]);
	}
    layer->loadWeighs(stream);
    fclose(stream);
    return layer;
}

Layer* createAndSaveLayer(Test* test, Buffer** controlInputs)
{
    Layer* controlLayer = new Layer(size, test->getBufferType(), IDENTITY, C);

    for (unsigned i = 0; i < NUM_INPUTS; i++){
		controlLayer->addInput(controlInputs[i]);
	}
    controlLayer->randomWeighs(initialWeighsRange);

	FILE* stream = fopen(PATH, "w+b");
	controlLayer->save(stream);
	controlLayer->saveWeighs(stream);
	fclose(stream);
    return controlLayer;
}

unsigned testCalculateOutput(Test* test)
{
	START_TEST

	Buffer* controlInputBuffers[NUM_INPUTS];
	for (unsigned i = 0; i < NUM_INPUTS; i++) {
		BufferType bufferTypeAux = BYTE;
		while (bufferTypeAux == BYTE) {
			bufferTypeAux = (BufferType)Random::positiveInteger(BUFFER_TYPE_DIM);
		}
		controlInputBuffers[i] = Factory::newBuffer(size, bufferTypeAux, C);
		controlInputBuffers[i]->random(initialWeighsRange);
	}

	Layer* controlLayer = createAndSaveLayer(test, controlInputBuffers);
	controlLayer->calculateOutput();

	Buffer* inputBuffers[NUM_INPUTS];
	for (unsigned i = 0; i < NUM_INPUTS; i++) {
		inputBuffers[i] = Factory::newBuffer(size, controlInputBuffers[i]->getBufferType(), test->getImplementationType());
		inputBuffers[i]->copyFrom(controlInputBuffers[i]);
	}

	Layer* layer = createAndLoadLayer(test->getImplementationType(), inputBuffers);

    //test calculation
	layer->calculateOutput();

    differencesCounter += Test::assertEquals(controlLayer->getOutput(), layer->getOutput());

	delete (layer);
	for (unsigned i = 0; i < NUM_INPUTS; i++) {
		delete(inputBuffers[i]);
	}
	delete (controlLayer);
	for (unsigned i = 0; i < NUM_INPUTS; i++) {
		delete(controlInputBuffers[i]);
	}

	END_TEST
}

int main(int argc, char *argv[]) {
	Chronometer total;
	Test test;
	total.start();

	test.addIterator(&size, 1, 50, 49);
	test.with(ET_IMPLEMENTATION, 1, SSE2);
	test.exclude(ET_BUFFER, 1, BYTE);
	test.printParameters();

	try {
		test.test(testCalculateOutput, "Layer::calculateOutput");

		printf("Exit success.\n");
		MemoryManagement::printTotalAllocated();
		MemoryManagement::printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
//	} catch (...) {
//		printf("An error was thrown.\n", 1);
	}

	//MemoryManagement::mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
