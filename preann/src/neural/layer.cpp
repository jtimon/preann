#include "layer.h"

ImplementationType Layer::getImplementationType()
{
    return output->getImplementationType();
}

Layer::Layer()
{
    this->functionType = FT_IDENTITY;
    output = NULL;
    thresholds = NULL;
    tOuputInterface = NULL;
}

Layer::Layer(unsigned size, BufferType outputType, FunctionType functionType,
             ImplementationType implementationType)
{
    this->functionType = functionType;
    output = Factory::newBuffer(size, outputType, implementationType);
    thresholds = Factory::newThresholds(output, implementationType);
    tOuputInterface = NULL;
}

Layer::Layer(FILE* stream, ImplementationType implementationType)
{
    fread(&functionType, sizeof(FunctionType), 1, stream);
    output = Factory::newBuffer(stream, implementationType);
    thresholds = Factory::newThresholds(output, implementationType);
    tOuputInterface = NULL;
}

void Layer::save(FILE* stream)
{
    fwrite(&functionType, sizeof(FunctionType), 1, stream);
    output->save(stream);
}

Layer::~Layer()
{
    CLEAR_PTR_VECTOR(Connection, connections)
    if (thresholds) {
        delete (thresholds);
    }
    if (output) {
        delete (output);
    }
    if (tOuputInterface) {
        delete (tOuputInterface);
    }
}

void Layer::loadWeighs(FILE* stream)
{
    unsigned numInputs;
    fread(&numInputs, sizeof(unsigned), 1, stream);
    if (numInputs != connections.size()) {
        std::string error = "Cannot load weighs: the layer doesn't have that numer of inputs.";
        throw error;
    }
    for (unsigned i = 0; i < connections.size(); i++) {
        connections[i]->load(stream);
    }
    thresholds->load(stream);
}

void Layer::saveWeighs(FILE* stream)
{
    unsigned numberInputs = connections.size();
    fwrite(&numberInputs, sizeof(unsigned), 1, stream);
    for (unsigned i = 0; i < connections.size(); i++) {
        connections[i]->save(stream);
    }
    thresholds->save(stream);
}

Interface* Layer::getOutputInterface()
{
    if (tOuputInterface == NULL) {
        tOuputInterface = new Interface(output->getSize(), output->getBufferType());
        output->copyToInterface(tOuputInterface);
    }
    return tOuputInterface;
}

void Layer::calculateOutput()
{
    if (!output) {
        std::string error = "Cannot calculate the output of a Layer without output.";
        throw error;
    }
    //TODO B do not use clone on the thresholds, compare with them in activation (one write less)
    //	Buffer* results = newBuffer(thresholds->getSize(), thresholds->getBufferType());
    Buffer* results = thresholds->clone();

    for (unsigned i = 0; i < connections.size(); i++) {
        connections[i]->calculateAndAddTo(results);
    }

    output->activation(results, functionType);
    //	thresholds->activation(results, functionType, output);
    if (tOuputInterface != NULL) {
        output->copyToInterface(tOuputInterface);
    }
    delete (results);
}

void Layer::addInput(Buffer* input)
{
    Connection* newConnection = Factory::newConnection(input, output->getSize());
    connections.push_back(newConnection);
}

void Layer::randomWeighs(float range)
{
    for (unsigned i = 0; i < connections.size(); i++) {
        Interface aux(connections[i]->getSize(), connections[i]->getBufferType());
        aux.random(range);
        connections[i]->copyFromInterface(&aux);
    }
    Interface aux(output->getSize(), BT_FLOAT);
    aux.random(range);
    thresholds->copyFromInterface(&aux);
}

unsigned Layer::getNumberInputs()
{
    return connections.size();
}

Buffer* Layer::getInput(unsigned pos)
{
    return connections[pos]->getInput();
}

Buffer* Layer::getOutput()
{
    return output;
}

Connection* Layer::getThresholds()
{
    return thresholds;
}

Connection* Layer::getConnection(unsigned inputPos)
{
    if (inputPos > connections.size()) {
        string error = "Cannot access the Connection in position " + to_string(inputPos)
                + ": the Layer has only " + to_string(connections.size()) + " inputs.";
        throw error;
    }
    return connections[inputPos];
}

FunctionType Layer::getFunctionType()
{
    return functionType;
}

void Layer::copyWeighs(Layer* sourceLayer)
{
    if (connections.size() != sourceLayer->getNumberInputs()) {
        std::string error = "Layer::copyWeighs : Cannot copyWeighs from a layer with "
                + to_string(sourceLayer->getNumberInputs()) + " connections to a layer with "
                + to_string(connections.size());
        throw error;
    }
    if (this->getImplementationType() != sourceLayer->getImplementationType()) {
        std::string error =
                "Layer::copyWeighs : The layers are incompatible: the implementation is different.";
        throw error;
    }
    //TODO L implementar metodo Buffer::copyFast restringido a bufferes con el mismo tipo de implementacion
    for (int i = 0; i < connections.size(); i++) {
        connections[i]->copyFrom(sourceLayer->getConnection(i));
    }
    thresholds->copyFrom(sourceLayer->getThresholds());
}

