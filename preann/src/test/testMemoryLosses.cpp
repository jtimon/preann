#include <iostream>
#include <fstream>

using namespace std;

#include "common/dummy.h"
#include "common/test.h"
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
    Buffer* buffer = Dummy::buffer(parametersMap);
    Layer* layer = Dummy::layer(parametersMap, buffer);

    delete (layer);
    delete (buffer);
}

void testNeuralNet(ParametersMap* parametersMap)
{
    Interface* input = Dummy::interface(parametersMap);
    NeuralNet* net = Dummy::neuralNet(parametersMap, input);

    delete (net);
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
    Task* task = new BinaryTask(BO_OR, size, 5);
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
        test.parameters.putNumber(Test::MEM_LOSSES, 0);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        RangeLoop* sizeLoop = new RangeLoop(Dummy::SIZE, 100, 101, 100);
        test.addLoop(sizeLoop);

        RangeLoop* outputSizeLoop = new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2);
        test.addLoop(outputSizeLoop);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER);
        test.addLoop(bufferTypeLoop);

//        test.addLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION));
        test.addLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION, 2, IT_C, IT_SSE2));

        test.getLoop()->print();

//        test.test(testLoops, &parametersMap, "test loops");

        test.test(testBuffer, "Buffer::memory_test");

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);

        test.test(testConnection, "Connection::memory_test");

        RangeLoop* numInputsLoop = new RangeLoop(Dummy::NUM_INPUTS, 1, 3, 1);
        test.addLoop(numInputsLoop);

        test.test(testLayer, "Layer::memory_test");

        RangeLoop* numLayersLoop = new RangeLoop(Dummy::NUM_LAYERS, 1, 3, 1);
        test.addLoop(numLayersLoop);

        test.test(testNeuralNet, "NeuralNet::memory_test");

        sizeLoop->resetRange(1, 3, 1);
        outputSizeLoop->resetRange(1, 1, 1);
        numInputsLoop->resetRange(1, 1, 1);
        numLayersLoop->resetRange(1, 1, 1);

        test.test(testPopulation, "Population::memory_test");

        MemoryManagement::printListOfPointers();

        printf("Exit success.\n", 1);
    } catch (string error) {
        cout << "Error: " << error << endl;
    } catch (...) {
        printf("An error was thrown.\n", 1);
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
