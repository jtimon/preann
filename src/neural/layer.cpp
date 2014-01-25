#include "layer.h"

ImplementationType Layer::getImplementationType()
{
    return output->getImplementationType();
}

unsigned Layer::getSize()
{
	return output->getSize();
}

BufferType Layer::getBufferType()
{
	return output->getBufferType();
}

Layer::Layer()
{
    this->functionType = FT_IDENTITY;
    results = NULL;
    thresholds = NULL;
    output = NULL;
    outputInterface = NULL;
}

Layer::Layer(unsigned size, BufferType outputType, FunctionType functionType,
             ImplementationType implementationType)
{
    this->functionType = functionType;
    results = Factory::newBuffer(size, BT_FLOAT, implementationType);
    thresholds = Factory::newConnection(results, 1);
    output = Factory::newBuffer(size, outputType, implementationType);
    outputInterface = NULL;
}

Layer::Layer(FILE* stream, ImplementationType implementationType)
{
    fread(&functionType, sizeof(FunctionType), 1, stream);
    output = Factory::newBuffer(stream, implementationType);
    results = Factory::newBuffer(output->getSize(), BT_FLOAT, implementationType);
    thresholds = Factory::newConnection(results, 1);
    outputInterface = NULL;
}

void Layer::save(FILE* stream)
{
    fwrite(&functionType, sizeof(FunctionType), 1, stream);
    output->save(stream);
}

Layer::~Layer()
{
    CLEAR_PTR_VECTOR(Connection, connections)
    if (results) {
        delete (results);
    }
    if (thresholds) {
        delete (thresholds);
    }
    if (output) {
        delete (output);
    }
    if (outputInterface) {
        delete (outputInterface);
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
    if (outputInterface == NULL) {
        outputInterface = new Interface(output->getSize(), output->getBufferType());
        output->copyToInterface(outputInterface);
    }
    return outputInterface;
}

void Layer::calculateOutput()
{
    if (!output) {
        std::string error = "Cannot calculate the output of a Layer without output.";
        throw error;
    }

    results->reset();
    for (unsigned i = 0; i < connections.size(); i++) {
        connections[i]->calculateAndAddTo(results);
    }
    thresholds->activation(output, functionType);

    if (outputInterface != NULL) {
        output->copyToInterface(outputInterface);
    }
}

void Layer::addInput(Layer* input)
{
	Util::check(getImplementationType() != input->getImplementationType(), 
			"Layer::addInput : layers of different implementation types are not compatible.");
	
    Connection* newConnection = Factory::newConnection(input->getOutput(), output->getSize());
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
	Util::check( inputPos > connections.size(), "Layer::getConnection : Cannot access the Connection in position " + to_string(inputPos)
            + ": the Layer has only " + to_string(connections.size()) + " inputs.");
    return connections[inputPos];
}

FunctionType Layer::getFunctionType()
{
    return functionType;
}

void Layer::copyWeighs(Layer* sourceLayer)
{
	Util::check(getImplementationType() != sourceLayer->getImplementationType(), 
			"Layer::copyWeighs : layers of different implementation types are not compatible.");
	
	Util::check(getSize() != sourceLayer->getSize(), "Layer::copyWeighs : Cannot copyWeighs from layers that have different size.");
	
	Util::check(getBufferType() != sourceLayer->getBufferType(), "Layer::copyWeighs : Cannot copyWeighs from layers that have different BufferType.");
	
	Util::check(connections.size() != sourceLayer->getNumberInputs(), 
			"Layer::copyWeighs : Cannot copyWeighs from a layer with "
			                + to_string(sourceLayer->getNumberInputs()) + " connections to a layer with "
			                + to_string(connections.size()));
	
    //TODO L implementar metodo Buffer::copyFast restringido a bufferes con el mismo tipo de implementacion
    for (int i = 0; i < connections.size(); i++) {
        connections[i]->copyFrom(sourceLayer->getConnection(i));
    }
    thresholds->copyFrom(sourceLayer->getThresholds());
}

