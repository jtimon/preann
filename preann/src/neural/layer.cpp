#include "layer.h"

ImplementationType Layer::getImplementationType()
{
	return thresholds->getImplementationType();
}

Buffer* Layer::newBuffer(FILE* stream)
{
	return  Factory::newBuffer(stream, getImplementationType());
}

Buffer* Layer::newBuffer(unsigned size, BufferType bufferType)
{
	return Factory::newBuffer(size, bufferType, getImplementationType());
}

Layer::Layer(unsigned size, BufferType outputType, FunctionType functionType, ImplementationType implementationType)
{
	connections = NULL;
	numberInputs = 0;
	this->functionType = functionType;
	output = Factory::newBuffer(size, outputType, implementationType);
	thresholds = Factory::newThresholds(output, implementationType);
}

Layer::Layer(FILE* stream, ImplementationType implementationType)
{
	connections = NULL;
	numberInputs = 0;
	fread(&functionType, sizeof(FunctionType), 1, stream);
	output = Factory::newBuffer(stream, implementationType);
	thresholds = Factory::newThresholds(output, implementationType);
}

void Layer::save(FILE* stream)
{
	fwrite(&functionType, sizeof(FunctionType), 1, stream);
	output->save(stream);
}

Layer::~Layer()
{
	if (connections) {
		for (unsigned i=0; i < numberInputs; i++){
			delete(connections[i]);
		}
		mi_free(connections);
	}
	if (thresholds) {
		delete(thresholds);
	}
	if (output) {
		delete (output);
	}
}

void Layer::loadWeighs(FILE* stream)
{
	unsigned numInputs;
	fread(&numInputs, sizeof(unsigned), 1, stream);
	if (numInputs != numberInputs){
		std::string error = "Cannot load weighs: the layer doesn't have that numer of inputs.";
		throw error;
	}
	for(unsigned i=0; i < numberInputs; i++){
		connections[i]->load(stream);
	}
	thresholds->load(stream);
}

void Layer::saveWeighs(FILE* stream)
{
	fwrite(&numberInputs, sizeof(unsigned), 1, stream);
	for(unsigned i=0; i < numberInputs; i++){
		connections[i]->save(stream);
	}
	thresholds->save(stream);
}


Interface *Layer::getOutputInterface()
{
	if (tOuputInterface == NULL){
		tOuputInterface = new Interface(output->getSize(), output->getBufferType());
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

	for(unsigned i=0; i < numberInputs; i++){
		connections[i]->calculateAndAddTo(results);
	}

	output->activation(results, functionType);
//	thresholds->activation(results, functionType, output);
	if (tOuputInterface != NULL){
		output->copyToInterface(tOuputInterface);
	}
	delete(results);
}

void Layer::addInput(Buffer* input)
{
	Connection* newConnection = Factory::newConnection(input, output->getSize(), getImplementationType());
	Connection** newConnections = (Connection**) mi_malloc(sizeof(Connection*) * (numberInputs + 1));

	if (connections) {
		memcpy(newConnections, connections, numberInputs * sizeof(Connection*));
		mi_free(connections);
	}

	connections = newConnections;
	connections[numberInputs] = newConnection;
	++numberInputs;
}

void Layer::randomWeighs(float range)
{
	for (unsigned i=0; i < numberInputs; i++){
		Interface aux(connections[i]->getSize(), connections[i]->getBufferType());
		aux.random(range);
		connections[i]->copyFromInterface(&aux);
	}
	Interface aux(output->getSize(), FLOAT);
	aux.random(range);
	thresholds->copyFromInterface(&aux);
}

unsigned Layer::getNumberInputs()
{
	return numberInputs;
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
	if (inputPos > numberInputs){
		char buffer[100];
		sprintf(buffer, "Cannot access the Connection in position %d: the Layer has only %d inputs.",
				inputPos, numberInputs);
		std::string error = buffer;
		throw error;
	}
	if (!connections){
		std::string error = "The layer has no connections.";
		throw error;
	}
	return connections[inputPos];
}

FunctionType Layer::getFunctionType()
{
   return functionType;
}

void Layer::copyWeighs(Layer* other)
{
	if(numberInputs != other->getNumberInputs()){
		char buffer[100];
		sprintf(buffer, "Cannot copyWeighs from a layer with %d connections to a layer with %d.",
				other->getNumberInputs(), numberInputs);
		std::string error = buffer;
		throw error;
	}
	if (this->getImplementationType() != other->getImplementationType()){
		std::string error = "The layers are incompatible: the implementation is different.";
		throw error;
	}
	//TODO L implementar metodo Buffer::copyFast restringido a bufferes con el mismo tipo de implementacion
	for(int i=0; i < numberInputs; i++){
		connections[i]->copyFrom(other->getConnection(i));
	}
	thresholds->copyFrom(other->getThresholds());
}

