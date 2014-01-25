#include <iostream>
#include <fstream>

using namespace std;

#include "loopTest/test.h"
#include "common/dummy.h"
#include "genetic/population.h"
#include "tasks/binaryTask.h"
#include "common/chronometer.h"

void testBuffer(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    delete (buffer);
//    unsigned* aa = (unsigned*)MemoryManagement::malloc(sizeof(unsigned) * 5);
}

void testConnection(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    Connection* connection = Dummy::connection(parametersMap, buffer);

    delete (connection);
    delete (buffer);
}

void testLayer(ParametersMap* parametersMap)
{
	ImplementationType implementationType = (ImplementationType) (parametersMap->getNumber(
	                Enumerations::enumTypeToString(ET_IMPLEMENTATION)));

	Interface* interfInput = Dummy::interface(parametersMap);
	Layer* input = new InputLayer(interfInput, implementationType);
    Layer* layer = Dummy::layer(parametersMap, input);

    delete (layer);
    delete (input);
    delete (interfInput);
}

void testNeuralNet(ParametersMap* parametersMap)
{
    Interface* input = Dummy::interface(parametersMap);
    NeuralNet* net = Dummy::neuralNet(parametersMap, input);

    delete (net);
    delete (input);
}

void testIndividual(ParametersMap* parametersMap)
{
    Interface* input = Dummy::interface(parametersMap);
    Individual* individual = Dummy::individual(parametersMap, input);

    delete (individual);
    delete (input);
}

void testPopulation(ParametersMap* parametersMap)
{
    BufferType bufferType = (BufferType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    ImplementationType implementationType = (ImplementationType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    FunctionType functionType = (FunctionType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));

    unsigned size = (unsigned) parametersMap->getNumber(Dummy::SIZE);

    Interface* input = new Interface(size, bufferType);
    Task* task = new BinaryTask(BO_OR, bufferType, size, 5);
    Individual* example = new Individual(implementationType);
    example->addInputLayer(input);
    example->addInputLayer(input);
    example->addInputLayer(input);
    example->addLayer(size, bufferType, functionType);
    example->addLayer(size, bufferType, functionType);
    example->addLayer(size, bufferType, functionType);
    example->addInputConnection(0, 0);
    example->addInputConnection(1, 0);
    example->addInputConnection(2, 0);
    example->addLayersConnection(0, 1);
    example->addLayersConnection(0, 2);
    example->addLayersConnection(1, 2);
    example->addLayersConnection(2, 0);
    Population* population = new Population(task, example, 5, 20);

    delete (population);
    delete (example);
    delete (task);
    delete (input);
}

void testLoops(ParametersMap* parametersMap)
{
    parametersMap->print();
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Test test;
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        RangeLoop* sizeLoop = new RangeLoop(Dummy::SIZE, 100, 101, 100);
        Loop* loop = sizeLoop;

        RangeLoop* outputSizeLoop = new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2);
        loop->addInnerLoop(outputSizeLoop);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER);
        loop->addInnerLoop(bufferTypeLoop);

//        loop->addInnerLoop(new EnumLoop(ET_IMPLEMENTATION, 2, IT_C, IT_SSE2));
        sizeLoop->addInnerLoop(new EnumLoop(ET_IMPLEMENTATION));


        loop->print();

//        test.testMemoryLosses(testLoops, &parametersMap, "test loops", &loop);

        test.testMemoryLosses(testBuffer, "Buffer", loop);

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);

        test.testMemoryLosses(testConnection, "Connection", loop);

        RangeLoop* numInputsLoop = new RangeLoop(Dummy::NUM_INPUTS, 1, 3, 1);
        loop->addInnerLoop(numInputsLoop);

        test.testMemoryLosses(testLayer, "Layer", loop);

        RangeLoop* numLayersLoop = new RangeLoop(Dummy::NUM_LAYERS, 1, 3, 1);
        loop->addInnerLoop(numLayersLoop);

        test.testMemoryLosses(testNeuralNet, "NeuralNet", loop);
        test.testMemoryLosses(testIndividual, "Individual", loop);

        sizeLoop->resetRange(1, 3, 1);
        outputSizeLoop->resetRange(1, 1, 1);
        numInputsLoop->resetRange(1, 1, 1);
        numLayersLoop->resetRange(1, 1, 1);

        test.testMemoryLosses(testPopulation, "Population", loop);

        delete(loop);

        MemoryManagement::printListOfPointers();

        printf("Exit success.\n");
    } catch (string& error) {
        cout << "Error: " << error << endl;
    } catch (...) {
        printf("An error was thrown.\n");
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
