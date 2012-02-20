#include <iostream>
#include <fstream>

using namespace std;

#include "loop.h"
#include "dummy.h"
#include "population.h"
#include "binaryTask.h"
#include "chronometer.h"

unsigned memoryLosses = 0;

void checkAndPrintErrors(string testingClass, ParametersMap* parametersMap)
{
    if (MemoryManagement::getPtrCounter() > 0
            || MemoryManagement::getTotalAllocated() > 0) {

        string state = parametersMap->getString(LOOP_STATE);

        cout << "Memory loss detected testing class " << testingClass
                << " at state " << state << ".\n" << endl;

        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
        MemoryManagement::clear();
        memoryLosses++;
    }
}

void testBuffer(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    delete (buffer);
//    unsigned* aa = (unsigned*)MemoryManagement::malloc(sizeof(unsigned) * 5);

    checkAndPrintErrors("Buffer", parametersMap);
}

void testConnection(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    Connection* connection = Dummy::connection(parametersMap, buffer);

    delete (connection);
    delete (buffer);

    checkAndPrintErrors("Connection", parametersMap);
}

void testLayer(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    Layer* layer = Dummy::layer(parametersMap, buffer);

    delete (layer);
    delete (buffer);

    checkAndPrintErrors("Layer", parametersMap);
}

void testNeuralNet(ParametersMap* parametersMap)
{
    Interface* input = Dummy::interface(parametersMap);
    NeuralNet* net = Dummy::neuralNet(parametersMap, input);

    delete (net);
    delete (input);

    checkAndPrintErrors("NeuralNet", parametersMap);
}

void testPopulation(ParametersMap* parametersMap)
{
    BufferType bufferType = (BufferType)parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_BUFFER));
    ImplementationType implementationType =
            (ImplementationType)parametersMap->getNumber(
                    Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    FunctionType functionType = (FunctionType)parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));

    unsigned size = (unsigned)parametersMap->getNumber("size");

    Interface* input = new Interface(size, bufferType);
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
    Task* task = new BinaryTask(BO_OR, size, 5);
    Population* population = new Population(task, example, 5, 20);

    delete (population);
    delete (example);
    delete (task);
    delete (input);
    checkAndPrintErrors("Population", parametersMap);
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
        Loop* loop;
        ParametersMap parametersMap;
        parametersMap.putNumber("initialWeighsRange", 20);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION),
                FT_IDENTITY);

        RangeLoop* sizeLoop = new RangeLoop("size", 100, 101, 100, NULL);
        loop = sizeLoop;

        RangeLoop* outputSizeLoop = new RangeLoop("outputSize", 1, 4, 2, loop);
        loop = outputSizeLoop;

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(
                ET_BUFFER), ET_BUFFER, loop);
        loop = bufferTypeLoop;

        loop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                ET_IMPLEMENTATION, loop);

        loop->print();

//        loop->repeatFunction(testLoops, &parametersMap, "test loops");

        loop->repeatFunction(testBuffer, &parametersMap, "Buffer::memory_test");

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        loop->print();

        loop->repeatFunction(testConnection, &parametersMap, "Connection::memory_test");

        RangeLoop* numInputsLoop = new RangeLoop("numInputs", 1, 3, 1, loop);
        loop = numInputsLoop;
        loop->print();

        loop->repeatFunction(testLayer, &parametersMap, "Layer::memory_test");

        RangeLoop* numLayersLoop = new RangeLoop("numLayers", 1, 3, 1, loop);
        loop = numLayersLoop;
        loop->print();

        loop->repeatFunction(testNeuralNet, &parametersMap, "NeuralNet::memory_test");

        sizeLoop->resetRange(1, 3, 1);
        outputSizeLoop->resetRange(1, 1, 1);
        numInputsLoop->resetRange(1, 1, 1);
        numLayersLoop->resetRange(1, 1, 1);
        loop->print();

        loop->repeatFunction(testPopulation, &parametersMap, "Population::memory_test");

        delete (loop);

        printf("Exit success.\n", 1);
        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
    } catch (string error) {
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
