
#include <iostream>
#include <fstream>

using namespace std;

#include "test.h"
#include "population.h"
#include "binaryTask.h"
#include "chronometer.h"


unsigned memoryLosses = 0;

void checkAndPrintErrors(string testingClass, Test* test)
{
    if(MemoryManagement::getPtrCounter() > 0 || MemoryManagement::getTotalAllocated() > 0){
        cout << "Memory loss detected testing class " << testingClass << ".\n" << endl;
        test->printCurrentState();
        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
        memoryLosses++;
    }
}

void testBuffer(Test* test)
{
	START_BUFFER
	END_BUFFER

    checkAndPrintErrors("Buffer", test);
}

void testConnection(Test* test)
{
	START_CONNECTION
	END_CONNECTION

    checkAndPrintErrors("Connection", test);
}

void testLayer(Test* test)
{
    Layer *layer = new Layer(test->getIterValue("size"), (BufferType)test->getEnum(ET_BUFFER), FT_IDENTITY, (ImplementationType)test->getEnum(ET_IMPLEMENTATION));
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    layer->addInput(layer->getOutput());
    delete(layer);

    checkAndPrintErrors("Layer", test);
}

void testNeuralNet(Test* test)
{
	unsigned size = test->getIterValue("size");
    BufferType bufferType = (BufferType)test->getEnum(ET_BUFFER);
    ImplementationType implementationType = (ImplementationType)test->getEnum(ET_IMPLEMENTATION);

    NeuralNet* net = new NeuralNet(implementationType);
    Interface* input = new Interface(size, bufferType);
	net->addInputLayer(input);
	net->addInputLayer(input);
	net->addInputLayer(input);
	net->addLayer(size, bufferType, FT_IDENTITY);
	net->addLayer(size, bufferType, FT_IDENTITY);
	net->addLayer(size, bufferType, FT_IDENTITY);
	net->addInputConnection(0, 0);
	net->addInputConnection(1, 0);
	net->addInputConnection(2, 0);
	net->addLayersConnection(0, 1);
	net->addLayersConnection(0, 2);
	net->addLayersConnection(1, 2);
	net->addLayersConnection(2, 0);

    delete(net);
    delete(input);
    checkAndPrintErrors("NeuralNet", test);
}

void testPopulation(Test* test)
{
	GET_SIZE

    BufferType bufferType = (BufferType)test->getEnum(ET_BUFFER);
    ImplementationType implementationType = (ImplementationType)test->getEnum(ET_IMPLEMENTATION);

    Interface* input = new Interface(size, bufferType);
    Individual* example = new Individual(implementationType);
	example->addInputLayer(input);
	example->addInputLayer(input);
	example->addInputLayer(input);
	example->addLayer(size, bufferType, FT_IDENTITY);
	example->addLayer(size, bufferType, FT_IDENTITY);
	example->addLayer(size, bufferType, FT_IDENTITY);
	example->addInputConnection(0, 0);
	example->addInputConnection(1, 0);
	example->addInputConnection(2, 0);
	example->addLayersConnection(0, 1);
	example->addLayersConnection(0, 2);
	example->addLayersConnection(1, 2);
	example->addLayersConnection(2, 0);
    Task* task = new BinaryTask(BO_OR, size, 5);
    Population* population = new Population(task, example, 5, 20);

    delete(population);
    delete(example);
    delete(task);
    delete(input);
    checkAndPrintErrors("Population", test);
}

//int a, b, c, d;
//unsigned miFuncioncita(Test* test){
//	test->printCurrentState();
//	return Random::positiveInteger(2);
//}

int main(int argc, char *argv[]) {

	Test test;
	Chronometer total;
	total.start();
	try {
//		test.withAll(ET_BUFFER);
//		test.withAll(ET_IMPLEMENTATION);
//		test.withAll(ET_CROSS_ALG);
//		test.withAll(ET_CROSS_LEVEL);
//		test.withAll(ET_MUTATION_ALG);
//		test.withAll(ET_FUNCTION);
//		test.addIterator(&a, 1, 2, 1);
//		test.addIterator(&b, 1, 2, 1);
//		test.addIterator(&c, 1, 2, 1);
//		test.addIterator(&d, 1, 2, 1);
//		test.test(miFuncioncita, "afdgfdgd");

		test.addIterator("size", 100, 101, 100);
		test.addIterator("outputSize", 1, 4, 2);
		float initialWeighsRange = 20;
		test.putVariable("initialWeighsRange", &initialWeighsRange);
		test.withAll(ET_BUFFER);
		test.withAll(ET_IMPLEMENTATION);
		test.printParameters();

		test.simpleTest(testBuffer, "Buffer::memory");

		test.exclude(ET_BUFFER, 1, BT_BYTE);
		test.printParameters();

		test.simpleTest(testConnection, "Connection::memory");
		test.simpleTest(testLayer, "Layer::memory");
		test.simpleTest(testNeuralNet, "NeuralNet::memory");
		test.simpleTest(testPopulation, "Population::memory");

		printf("Exit success.\n", 1);
		MemoryManagement::printTotalAllocated();
		MemoryManagement::printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
	} catch (...) {
		printf("An error was thrown.\n", 1);
	}

	cout << "Total memory losses: " << memoryLosses << endl;
	MemoryManagement::printListOfPointers();

	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
