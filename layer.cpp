#include "layer.h"
#include "factory.h"

Vector* Layer::newVector(FILE* stream)
{
	return  Factory::newVector(stream, tImplementationType);
}

Vector* Layer::newVector(unsigned size, VectorType vectorType)
{
	return Factory::newVector(size, vectorType, tImplementationType);
}

Layer::Layer(unsigned size, VectorType outputType, FunctionType functionType, ImplementationType implementationType)
{
	tImplementationType = implementationType;

	connections = NULL;
	numberInputs = 0;
	this->functionType = functionType;
	output = Factory::newVector(size, outputType, implementationType);
	thresholds = Factory::newVector(size, FLOAT, implementationType);
}

Layer::Layer(FILE* stream, ImplementationType implementationType)
{
	tImplementationType = implementationType;

	fread(&functionType, sizeof(FunctionType), 1, stream);
	thresholds = Factory::newVector(stream, implementationType);
	output = Factory::newVector(stream, implementationType);

	fread(&numberInputs, sizeof(unsigned), 1, stream);
	connections = (Connection**) mi_malloc(numberInputs * sizeof(Connection*));

	for(unsigned i=0; i < numberInputs; i++){
		//TODO E esto puede llevar al pete
		connections[i] = Factory::newConnection(stream, output->getSize(), implementationType);
	}
}

void Layer::save(FILE* stream)
{
	fwrite(&functionType, sizeof(FunctionType), 1, stream);
	Factory::saveVector(thresholds, stream);
	Factory::saveVector(output, stream);

	fwrite(&numberInputs, sizeof(unsigned), 1, stream);
	for(unsigned i=0; i < numberInputs; i++){
		connections[i]->save(stream);
	}
}

Layer::~Layer()
{
	if (connections) {
		for (unsigned i=0; i < numberInputs; i++){
			delete(connections[i]);
		}
		mi_free(connections);
		connections = NULL;
	}
	if (thresholds) {
		delete(thresholds);
		thresholds = NULL;
	}
	if (output) {
		delete (output);
		output = NULL;
	}
}

void Layer::checkCompatibility(Layer* layer)
{
	if (this->getImplementationType() != layer->getImplementationType()){
		std::string error = "The layers are incompatible: the implementation is different.";
		throw error;
	}
	if (this->getOutput()->getSize() != layer->getOutput()->getSize()){
		std::string error = "The layers are incompatible: the output size is different.";
		throw error;
	}
	if (this->getOutput()->getVectorType() != layer->getOutput()->getVectorType()){
		std::string error = "The layers are incompatible: the output type is different.";
		throw error;
	}
	if (this->getNumberInputs() != layer->getNumberInputs()){
		std::string error = "The layers are incompatible: the number of inputs is different.";
		throw error;
	}
	for (unsigned i=0; i < numberInputs; i++){
		if (this->getInput(i)->getSize() != layer->getInput(i)->getSize()){
			std::string error = "The layers are incompatible: the size of an input is different.";
			throw error;
		}
		if (this->getInput(i)->getVectorType() != layer->getInput(i)->getVectorType()){
			std::string error = "The layers are incompatible: the type of an input is different.";
			throw error;
		}
	}
}

void Layer::calculateOutput()
{
	if (!output) {
		std::string error = "Cannot calculate the output of a Layer without output.";
		throw error;
	}
	//TODO B do not use clone on the thresholds, compare with them in activation (one write less)
	Vector* results = thresholds->clone();

	for(unsigned i=0; i < numberInputs; i++){
		connections[i]->addToResults(results);
	}

	output->activation(results, functionType);
	delete(results);
}

void Layer::addInput(Vector* input)
{
	//TODO T probar qué sucede con varios tipos de entrada
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

void Layer::setInput(Vector* input, unsigned pos)
{
	if (pos >= numberInputs){
		char buffer[100];
		sprintf(buffer, "Cannot set the input in position %d: the layer just have %d inputs.", pos, numberInputs);
		std::string error = buffer;
		throw error;
	}
	switch (input->getVectorType()){
	case BYTE:
		{
		std::string error = "Layer::setInput is not implemented for an input Vector of the VectorType BYTE";
		throw error;
		}
	default:
		break;
	}
	connections[pos]->setInput(input);
}

void Layer::randomWeighs(float range)
{
	Interface* aux = new Interface(output->getSize(), FLOAT);
	aux->random(range);
	thresholds->copyFrom(aux);
	delete(aux);

	for (unsigned i=0; i < numberInputs; i++){
		aux = new Interface(connections[i]->getSize(), connections[i]->getVectorType());
		aux->random(range);
		connections[i]->copyFrom(aux);
		delete(aux);
	}
}

void Layer::mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		std::string error = "Cannot mutate that output: the Layer hasn't so many neurons.";
		throw error;
	}
	if (inputLayer > numberInputs) {
		std::string error = "Cannot mutate that input: the Layer hasn't so many inputs.";
		throw error;
	}
	if (inputPos > connections[inputLayer]->getInput()->getSize()) {
		std::string error = "Cannot mutate that input: the input hasn't so many neurons.";
		throw error;
	}
	unsigned weighPos = (outputPos * connections[inputLayer]->getInput()->getSize()) + inputPos;
	connections[inputLayer]->mutate(weighPos, mutation);
}

void Layer::mutateThreshold(unsigned outputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		std::string error = "Cannot mutate that Threshold: the Layer hasn't so many neurons.";
		throw error;
	}
	thresholds->mutate(outputPos, mutation);
	//TODO L Layer::crossoverThreshold
}

void Layer::swapWeighs(Layer* layer)
{
	checkCompatibility(layer);

	unsigned size = numberInputs * sizeof(Vector*);
	Vector** temp = (Vector**) mi_malloc(size);
	memcpy(temp, connections, size);
	memcpy(connections, layer->connections, size);
	memcpy(layer->connections, temp, size);
}

unsigned Layer::getNumberInputs()
{
	return numberInputs;
}

Vector* Layer::getInput(unsigned pos)
{
	return connections[pos]->getInput();
}

Vector* Layer::getOutput()
{
	return output;
}

float* Layer::getThresholdsPtr()
{
	return (float*)thresholds->getDataPointer();
}

Connection* Layer::getConnection(unsigned inputPos)
{
	return connections[inputPos];
}

FunctionType Layer::getFunctionType()
{
   return functionType;
}


void Layer::copyWeighs(Layer* other)
{
	//TODO L implementar metodo
	std::string error = "CudaLayer::copyWeighs is not implemented.";
	throw error;
//	memcpy(thresholds, other->getThresholdsPtr(), output->getSize() * sizeof(float));
//	unsigned size;
//	if (inputType == FLOAT){
//		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
//	} else {
//		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
//	}
//	memcpy(weighs, other->getWeighsPtr(), size);
}

void Layer::mutateWeigh(float mutationRange)
{
	unsigned chosenInput = randomUnsigned(this->numberInputs);
	unsigned chosenOutput = randomUnsigned(output->getSize());
	unsigned chosenInputPos = randomUnsigned(connections[chosenInput]->getInput()->getSize());

	mutateWeigh(chosenOutput, chosenInput, chosenInputPos, randomFloat(mutationRange));
}

void Layer::mutateWeighs(float probability, float mutationRange)
{
	for (unsigned i=0; i < output->getSize(); i++){
		for (unsigned j=0; j < numberInputs; j++){
			for (unsigned k=0; k < connections[j]->getInput()->getSize(); k++){
				if (randomPositiveFloat(1) < probability) {
					mutateWeigh(i, j, k, randomFloat(mutationRange));
				}
			}
		}
		if (randomPositiveFloat(1) < probability) {
			mutateThreshold(i, randomFloat(mutationRange));
		}
	}
}

void Layer::crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector)
{
	connections[inputLayer]->crossover(other->getConnection(inputLayer), bitVector);
}

void Layer::crossoverInput(Layer *other, unsigned  inputLayer, Interface *bitVector)
{
	checkCompatibility(other);

	unsigned inputSize = connections[inputLayer]->getInput()->getSize();
	unsigned outputSize = output->getSize();
	Interface* inputBitVector = new Interface(inputSize * outputSize, BIT);

	for (unsigned i=0; i < inputSize; i++){
		if (bitVector->getElement(i)){
			for (unsigned j=0; j < outputSize; j++){
				unsigned offset = j * inputSize;
				inputBitVector->setElement(offset + i, 1);
			}
		}
	}
	crossoverWeighs(other, inputLayer, inputBitVector);
	delete bitVector;
}

void Layer::crossoverNeurons(Layer *other, Interface *bitVector)
{
	//TODO D comprobar esto en un nivel superior (o no comprobarlo) (o comprobarlo opcionalmente)
	checkCompatibility(other);

	unsigned outputSize = output->getSize();

	for (unsigned i=0; i < numberInputs; i++){

		unsigned inputSize = connections[i]->getInput()->getSize();
		Interface* inputBitVector = new Interface(inputSize * outputSize, BIT);

		for (unsigned j=0; j < outputSize; j++){
			if (bitVector->getElement(j)){
				unsigned offset = j * inputSize;
				for (unsigned k=0; k < inputSize; k++){
					inputBitVector->setElement(offset + k, 1);
				}
			}
		}
		crossoverWeighs(other, i, inputBitVector);
	}
	delete bitVector;
}
