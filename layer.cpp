#include "layer.h"
#include "factory.h"

Vector* Layer::newVector(FILE* stream)
{
	Interface* interface = new Interface();
	interface->load(stream);

	Vector* vector = Factory::newVector(interface->getSize(), interface->getVectorType(), getImplementationType());
	vector->copyFrom(interface);

	delete(interface);
	return  vector;
}

Vector* Layer::newVector(unsigned size, VectorType vectorType)
{
	return Factory::newVector(size, vectorType, getImplementationType());
}

Layer::Layer()
{
	inputs = NULL;
	connections = NULL;
	numberInputs = 0;
	thresholds = NULL;
	output = NULL;
}

Layer::~Layer()
{
}

void Layer::init(unsigned size, VectorType outputType, FunctionType functionType)
{
	this->functionType = functionType;
	output = newVector(size, outputType);
	thresholds = newVector(size, FLOAT);
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

	float* results = negativeThresholds();

	for(unsigned i=0; i < numberInputs; i++){
		inputCalculation(inputs[i], connections[i]->getDataPointer(), results);
	}
	output->activation(results, functionType);
}

void Layer::addInput(Vector* input)
{
	Vector* newWeighs;
	switch (input->getVectorType()){
	case BYTE:
		{
		std::string error = "Layer::addInput is not implemented for an input Vector of the VectorType BYTE";
		throw error;
		}
	case FLOAT:
		newWeighs = newVector(input->getSize() * output-> getSize(), FLOAT);
		break;
	case BIT:
	case SIGN:
		newWeighs = newVector(input->getSize() * output-> getSize(), BYTE);
		break;
	}
	//TODO probar qué sucede con varios tipos de entrada
	Vector** newInputs = (Vector**) mi_malloc(sizeof(Vector*) * (numberInputs + 1));
	Vector** newConnections = (Vector**) mi_malloc(sizeof(Vector*) * (numberInputs + 1));
	if (inputs) {
		memcpy(newInputs, inputs, numberInputs * sizeof(Vector*));
		memcpy(newConnections, connections, numberInputs * sizeof(Vector*));
		mi_free(inputs);
		mi_free(connections);
	}
	inputs = newInputs;
	connections = newConnections;

	inputs[numberInputs] = input;
	connections[numberInputs] = newWeighs;
	++numberInputs;
}

void Layer::setInput(Vector* input, unsigned pos)
{
	switch (input->getVectorType()){
	case BYTE:
		{
		std::string error = "Layer::setInput is not implemented for an input Vector of the VectorType BYTE";
		throw error;
		}
	default:
		break;
	}
	inputs[pos] = input;
}

void Layer::save(FILE* stream)
{
	fwrite(&functionType, sizeof(FunctionType), 1, stream);
	thresholds->save(stream);
	output->save(stream);

	fwrite(&numberInputs, sizeof(unsigned), 1, stream);
	for(unsigned i=0; i < numberInputs; i++){
		connections[i]->save(stream);
	}
}

void Layer::load(FILE* stream)
{
	fread(&functionType, sizeof(FunctionType), 1, stream);
	thresholds = newVector(stream);
	output = newVector(stream);

	fread(&numberInputs, sizeof(unsigned), 1, stream);
	inputs = (Vector**) mi_malloc(numberInputs * sizeof(Vector*));
	connections = (Vector**) mi_malloc(numberInputs * sizeof(Vector*));
	for(unsigned i=0; i < numberInputs; i++){
		//TODO esto puede llevar al pete
		inputs[i] = NULL;
		connections[i] = newVector(stream);
	}
}

void Layer::randomWeighs(float range)
{
	//TODO quitar esta comprobacion cuando deje de poder haber capas sin tamaño
	if (output == NULL){
		std::string error = "Cannot set random weighs to a layer with no output.";
		throw error;
	}

	Interface* aux = new Interface(output->getSize(), FLOAT);
	aux->random(range);
	thresholds->copyFrom(aux);
	delete(aux);


	for (unsigned i=0; i < numberInputs; i++){
		Vector* connection = connections[i];
		aux = new Interface(connection->getSize(), connection->getVectorType());
		aux->random(range);
		connection->copyFrom(aux);
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
	if (inputPos > inputs[inputLayer]->getSize()) {
		std::string error = "Cannot mutate that input: the input hasn't so many neurons.";
		throw error;
	}
	unsigned weighPos = (outputPos * inputs[inputLayer]->getSize()) + inputPos;
	connections[inputLayer]->mutate(weighPos, mutation);
}

void Layer::mutateThreshold(unsigned outputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		std::string error = "Cannot mutate that Threshold: the Layer hasn't so many neurons.";
		throw error;
	}
	thresholds->mutate(outputPos, mutation);
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
	return inputs[pos];
}

Vector* Layer::getOutput()
{
	return output;
}

float* Layer::getThresholdsPtr()
{
	return (float*)thresholds->getDataPointer();
}

Vector* Layer::getConnection(unsigned inputPos)
{
	return connections[inputPos];
}

FunctionType Layer::getFunctionType()
{
   return functionType;
}

/*
void Layer::copyWeighs(Layer* other)
{
	memcpy(thresholds, other->getThresholdsPtr(), output->getSize() * sizeof(float));
	unsigned size;
	if (inputType == FLOAT){
		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
	} else {
		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
	}
	memcpy(weighs, other->getWeighsPtr(), size);
}*/

void Layer::mutateWeigh(float mutationRange)
{
	unsigned chosenInput = randomUnsigned(this->numberInputs);
	unsigned chosenOutput = randomUnsigned(output->getSize());
	unsigned chosenInputPos = randomUnsigned(inputs[chosenInput]->getSize());

	mutateWeigh(chosenOutput, chosenInput, chosenInputPos, randomFloat(mutationRange));
}

void Layer::mutateWeighs(float probability, float mutationRange)
{
	for (unsigned i=0; i < output->getSize(); i++){
		for (unsigned j=0; j < numberInputs; j++){
			for (unsigned k=0; k < inputs[j]->getSize(); k++){
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

void Layer::crossoverInput(Layer *other, unsigned  inputLayer, Interface *bitVector)
{
	checkCompatibility(other);

	unsigned inputSize = inputs[inputLayer]->getSize();
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
	//TODO comprobar esto en un nivel superior (o no comprobarlo) (o comprobarlo opcionalmente)
	checkCompatibility(other);

	unsigned outputSize = output->getSize();

	for (unsigned i=0; i < numberInputs; i++){

		unsigned inputSize = inputs[i]->getSize();
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
