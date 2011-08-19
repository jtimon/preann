
#include <iostream>
#include <fstream>

using namespace std;

#include "population.h"
#include "chronometer.h"
#include "cuda_code.h"
#include "test.h"


#define PATH "/home/timon/layer.lay"

Test test;

unsigned char areEqual(float expected, float actual, BufferType bufferType)
{
	if (bufferType == FLOAT){
		return (expected - 1 < actual
			 && expected + 1 > actual);
	} else {
		return expected == actual;
	}
}

unsigned assertEquals(Buffer* expected, Buffer* actual)
{
    if(expected->getBufferType() != actual->getBufferType()){
        throw "The buffers are not even of the same type!";
    }
    if(expected->getSize() != actual->getSize()){
        throw "The buffers are not even of the same size!";
    }

	unsigned differencesCounter = 0;
	Interface* expectedInt = expected->toInterface();
	Interface* actualInt = actual->toInterface();

    for(unsigned i = 0;i < expectedInt->getSize();i++){
        if(!areEqual(expectedInt->getElement(i), actualInt->getElement(i), expectedInt->getBufferType())){
            printf("The buffers are not equal at the position %d (expected = %f actual %f).\n", i, expectedInt->getElement(i), actualInt->getElement(i));
            ++differencesCounter;
        }
    }
    delete(expectedInt);
	delete(actualInt);
	return differencesCounter;
}

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

Layer* createAndSaveLayer(unsigned size, BufferType bufferType, Buffer** controlInputs)
{
    Layer* controlLayer = new Layer(size, bufferType, IDENTITY, C);

    for (unsigned i = 0; i < NUM_INPUTS; i++){
		controlLayer->addInput(controlInputs[i]);
	}
    controlLayer->randomWeighs(test.getInitialWeighsRange());

	FILE* stream = fopen(PATH, "w+b");
	controlLayer->save(stream);
	controlLayer->saveWeighs(stream);
	fclose(stream);
    return controlLayer;
}

#define SIZE_MIN 1
#define SIZE_MAX 50
#define SIZE_INC 50

int main(int argc, char *argv[]) {
	Chronometer total;
	total.start();

	test.setInitialWeighsRange(20);
	test.setMaxSize(50);
	test.setIncSize(50);
	test.disableBufferType(BYTE);
	test.printParameters();

	try {
		for (test.sizeToMin(); test.hasNextSize(); test.sizeIncrement()) {
			for (test.bufferTypeToMin(); test.hasNextBufferType(); test.bufferTypeIncrement() ) {
				Buffer* controlInputBuffers[BUFFER_TYPE_DIM];
				for (unsigned i = 0; i < NUM_INPUTS; i++) {
					BufferType bufferTypeAux = BYTE;
					while (bufferTypeAux == BYTE) {
						bufferTypeAux = (BufferType)Random.positiveInteger(BUFFER_TYPE_DIM);
					}
					controlInputBuffers[i] = Factory::newBuffer(test.getSize(), bufferTypeAux, C);
					controlInputBuffers[i]->random(test.getInitialWeighsRange());
				}

			    Layer* controlLayer = createAndSaveLayer(test.getSize(), test.getBufferType(), controlInputBuffers);
			    controlLayer->calculateOutput();

			    for (test.implementationTypeToMin(); test.hasNextImplementationType(); test.implementationTypeIncrement()) {

					test.printCurrentState();

					Buffer* inputBuffers[BUFFER_TYPE_DIM];
					for (unsigned i = 0; i < NUM_INPUTS; i++) {
						inputBuffers[i] = Factory::newBuffer(test.getSize(), controlInputBuffers[i]->getBufferType(), test.getImplementationType());
						inputBuffers[i]->copyFrom(controlInputBuffers[i]);
					}

					Layer* layer = createAndLoadLayer(test.getImplementationType(), inputBuffers);

				    //test calculation
					layer->calculateOutput();

				    unsigned differences = Test::assertEquals(controlLayer->getOutput(), layer->getOutput());
				    if (differences != 0)
				    	printf("Errors on outputs: %d \n", differences);

					//test Weighs
				    for(unsigned i = 0; i < NUM_INPUTS; i++){
				        Connection* expectedWeighs = controlLayer->getConnection(i);
				        Connection* actualWeighs = layer->getConnection(i);
				        differences = Test::assertEquals(expectedWeighs, actualWeighs);
				        if (differences != 0)
				        	printf("Errors on weighs (input %d): %d \n", i, differences);
				    }
					delete (layer);
					for (unsigned i = 0; i < NUM_INPUTS; i++) {
						delete(inputBuffers[i]);
					}
				}
			    delete (controlLayer);
				for (unsigned i = 0; i < NUM_INPUTS; i++) {
					delete(controlInputBuffers[i]);
				}
			}
		}
		printf("Exit success.\n");
		MemoryManagement.printTotalAllocated();
		MemoryManagement.printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
//	} catch (...) {
//		printf("An error was thrown.\n", 1);
	}

	//MemoryManagement.mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
