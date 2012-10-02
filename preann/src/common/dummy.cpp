/*
 * dummy.cpp
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#include "dummy.h"
#include "tasks/binaryTask.h"
#include "tasks/reversiTask.h"

const string Dummy::SIZE = "Tamaño";
const string Dummy::WEIGHS_RANGE = "__initialWeighsRange";
const string Dummy::OUTPUT_SIZE = "__outputSize";
const string Dummy::NUM_INPUTS = "__numInputs";
const string Dummy::NUM_LAYERS = "__numLayers";
const string Dummy::NUM_TESTS = "__numTests";

Interface* Dummy::interface(ParametersMap* parametersMap)
{
    BufferType bufferType = (BufferType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER));

    unsigned size = (unsigned) parametersMap->getNumber(Dummy::SIZE);
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

    Interface* interface = new Interface(size, bufferType);
    interface->random(initialWeighsRange);

    return interface;
}

Buffer* Dummy::buffer(ParametersMap* parametersMap)
{
    BufferType bufferType = (BufferType) (parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER)));
    ImplementationType implementationType = (ImplementationType) (parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION)));

    unsigned size = (unsigned) (parametersMap->getNumber(Dummy::SIZE));
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

    Buffer* buffer = Factory::newBuffer(size, bufferType, implementationType);
    buffer->random(initialWeighsRange);

    return buffer;
}

Connection* Dummy::connection(ParametersMap* parametersMap, Buffer* buffer)
{
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);
    unsigned outputSize = parametersMap->getNumber(Dummy::OUTPUT_SIZE);

    Connection* connection = Factory::newConnection(buffer, outputSize);
    connection->random(initialWeighsRange);

    return connection;
}

Layer* Dummy::layer(ParametersMap* parametersMap, Layer* input)
{
    BufferType bufferType = (BufferType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    ImplementationType implementationType = (ImplementationType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    FunctionType functionType = (FunctionType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));

    unsigned size = (unsigned) parametersMap->getNumber(Dummy::SIZE);
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);
    unsigned numInputs = (unsigned) parametersMap->getNumber(Dummy::NUM_INPUTS);

    Layer* layer = new Layer(size, bufferType, functionType, implementationType);
    for (unsigned i = 0; i < numInputs; ++i) {
        layer->addInput(input);
    }
    layer->randomWeighs(initialWeighsRange);

    return layer;
}

void Dummy::addConnections(NeuralNet* net, Interface* input, unsigned numInputs, unsigned numLayers,
                           unsigned size, BufferType bufferType, FunctionType functionType)
{
    for (unsigned i = 0; i < numInputs; ++i) {
        net->addInputLayer(input);
    }
    for (unsigned j = 0; j < numLayers; ++j) {
        net->addLayer(size, bufferType, functionType);
    }
    for (unsigned i = 0; i < numInputs; ++i) {
        for (unsigned j = 0; j < numLayers; ++j) {
            net->addInputConnection(i, j);
        }
    }

    for (unsigned i = 0; i < numLayers; ++i) {
        for (unsigned j = 0; j < numLayers; ++j) {
            net->addLayersConnection(i, j);
        }
    }
}

NeuralNet* Dummy::neuralNet(ParametersMap* parametersMap, Interface* input)
{
    BufferType bufferType = (BufferType) (parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER)));
    ImplementationType implementationType = (ImplementationType) (parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION)));
    FunctionType functionType = (FunctionType) (parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION)));

    unsigned size = (unsigned) (parametersMap->getNumber(Dummy::SIZE));
    unsigned numInputs = (unsigned) (parametersMap->getNumber(Dummy::NUM_INPUTS));
    unsigned numLayers = parametersMap->getNumber(Dummy::NUM_LAYERS);

    NeuralNet* net = new NeuralNet(implementationType);

    addConnections(net, input, numInputs, numLayers, size, bufferType, functionType);

    return net;
}

Individual* Dummy::individual(ParametersMap* parametersMap, Interface* input)
{
    BufferType bufferType = (BufferType) (parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER)));
    ImplementationType implementationType = (ImplementationType) (parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION)));
    FunctionType functionType = (FunctionType) (parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION)));

    unsigned size = (unsigned) (parametersMap->getNumber(Dummy::SIZE));
    unsigned numInputs = (unsigned) (parametersMap->getNumber(Dummy::NUM_INPUTS));
    unsigned numLayers = parametersMap->getNumber(Dummy::NUM_LAYERS);

    Individual* individual = new Individual(implementationType);
    individual->setFitness(1);

    addConnections(individual, input, numInputs, numLayers, size, bufferType, functionType);

    return individual;
}

Task* Dummy::task(ParametersMap* parametersMap)
{
    Task* task;
    BufferType bufferType = (BufferType) (parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER)));
    TestTask testTask = (TestTask) parametersMap->getNumber(Enumerations::enumTypeToString(ET_TEST_TASKS));
    unsigned size = (unsigned) (parametersMap->getNumber(Dummy::SIZE));
    unsigned numTest;

    switch (testTask) {
        case TT_BIN_OR:
            task = new BinaryTask(BO_OR, bufferType, size);
            break;
        case TT_BIN_AND:
            task = new BinaryTask(BO_AND, bufferType, size);
            break;
        case TT_BIN_XOR:
            task = new BinaryTask(BO_XOR, bufferType, size);
            break;
        case TT_REVERSI:
            numTest = (unsigned) (parametersMap->getNumber(Dummy::NUM_TESTS));
            task = new ReversiTask(size, bufferType, numTest);
            break;
        default:
            string error = "Dummy::task not suported task " + to_string(testTask);
            throw error;
    }

    return task;
}

