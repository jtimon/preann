/*
 * dummy.cpp
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#include "dummy.h"

Interface* Dummy::interface(ParametersMap* parametersMap)
{
    BufferType bufferType = (BufferType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER));

    unsigned size = (unsigned) parametersMap->getNumber("size");
    float initialWeighsRange = parametersMap->getNumber(Factory::WEIGHS_RANGE);

    Interface* interface = new Interface(size, bufferType);
    interface->random(initialWeighsRange);

    return interface;
}

Layer* Dummy::layer(ParametersMap* parametersMap, Buffer* input)
{
    BufferType bufferType = (BufferType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    ImplementationType implementationType =
            (ImplementationType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    FunctionType functionType =
            (FunctionType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));

    unsigned size = (unsigned) parametersMap->getNumber("size");
    float initialWeighsRange = parametersMap->getNumber(Factory::WEIGHS_RANGE);
    unsigned numInputs = (unsigned) parametersMap->getNumber("numInputs");

    Layer* layer = new Layer(size, bufferType, functionType, implementationType);
    for (unsigned i = 0; i < numInputs; ++i) {
        layer->addInput(input);
    }
    layer->randomWeighs(initialWeighsRange);

    return layer;
}

NeuralNet* Dummy::neuralNet(ParametersMap* parametersMap, Interface* input)
{
    BufferType bufferType = (BufferType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    ImplementationType implementationType =
            (ImplementationType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    FunctionType functionType =
            (FunctionType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));

    unsigned size = (unsigned) parametersMap->getNumber("size");
    unsigned numInputs = (unsigned) parametersMap->getNumber("numInputs");
    unsigned numLayers = parametersMap->getNumber("numLayers");

    NeuralNet* net = new NeuralNet(implementationType);

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

    return net;
}

