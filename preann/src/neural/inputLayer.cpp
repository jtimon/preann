#include "inputLayer.h"
#include "factory/factory.h"

InputLayer::InputLayer(Interface* interface, ImplementationType implementationType)
{
    tInput = interface;
    output = Factory::newBuffer(interface->getSize(), interface->getBufferType(), implementationType);
    //TODO un solo interfaz para entrada y salida
    outputInterface = NULL;

    thresholds = NULL;
    functionType = FT_IDENTITY;
}

InputLayer::InputLayer(FILE* stream, ImplementationType implementationType)
{
    output = Factory::newBuffer(stream, implementationType);
    tInput = new Interface(output->getSize(), output->getBufferType());
    outputInterface = NULL;
    thresholds = NULL;
    functionType = FT_IDENTITY;
}

void InputLayer::save(FILE* stream)
{
    output->save(stream);
}

InputLayer::~InputLayer()
{
}

ImplementationType InputLayer::getImplementationType()
{
    return output->getImplementationType();
}

void InputLayer::addInput(Layer* input)
{
    std::string error = "addInput method does not work for InputLayer.";
    throw error;
}

Connection* InputLayer::getThresholds()
{
    std::string error = "getThresholds method does not work for InputLayer.";
    throw error;
    return NULL;
}

void InputLayer::calculateOutput()
{
    output->copyFromInterface(tInput);
}

void InputLayer::randomWeighs(float range)
{
    //do nothing
}

void InputLayer::copyWeighs(Layer* sourceLayer)
{
    if (connections.size() != sourceLayer->getNumberInputs()) {
        std::string error = "InputLayer::copyWeighs : Cannot copyWeighs from a layer with "
                + to_string(sourceLayer->getNumberInputs()) + " connections to a layer with "
                + to_string(connections.size());
        throw error;
    }
    if (this->getImplementationType() != sourceLayer->getImplementationType()) {
        std::string error =
                "InputLayer::copyWeighs : The layers are incompatible: the implementation is different.";
        throw error;
    }
    if (sourceLayer->getThresholds() != NULL) {
        std::string error = "InputLayer::copyWeighs : trying to copy from a non input layer";
        throw error;
    }
}

Interface* InputLayer::getInputInterface()
{
    return tInput;
}

