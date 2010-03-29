#include "layer.h"
/*
Layer::Layer()
{
	inputs = NULL;
	numberInputs = 0;
	totalWeighsPerOutput = 0;

	weighs = NULL;
	thresholds = NULL;

	output = NULL;

	inputType = FLOAT;
	outputType = FLOAT;
	functionType = IDENTITY;
}*/

Layer::Layer(VectorType inputType, VectorType outputType, FunctionType functionType)
{
	inputs = NULL;
	numberInputs = 0;
	totalWeighsPerOutput = 0;

	weighs = NULL;
	thresholds = NULL;

	output = NULL;

	this->inputType = inputType;
	this->outputType = outputType;
	switch (outputType){
		case FLOAT:
			this->functionType = functionType;
			break;
		case BIT:
			this->functionType = BINARY_STEP;
			break;
		case SIGN:
			this->functionType = BIPOLAR_STEP;
	}
}

Layer::~Layer()
{

}

void Layer::randomWeighs(float range)
{
	std::string error = "randomWeighs is not implemented.";
	throw error;
}

void Layer::addInput(Vector* input)
{
	if (input->getVectorType() != inputType) {
		string error = "Trying to add an incorrect type input.";
		error += " Layer inputs type: ";
		switch (inputType) {
			case FLOAT: error += "FLOAT";
				break;
			case BIT: error += "BIT";
				break;
			case SIGN: error += "SIGN";
				break;
		}
		error += " Input type: ";
		switch (input->getVectorType()) {
			case FLOAT: error += "FLOAT";
				break;
			case BIT: error += "BIT";
				break;
			case SIGN: error += "SIGN";
				break;
		}
		throw error;
	}

	Vector** newInputs = (Vector**) mi_malloc(sizeof(Vector*) * (numberInputs + 1));
	if (inputs) {
		memcpy(newInputs, inputs, numberInputs * sizeof(Vector*));
		mi_free(inputs);
	}
	inputs = newInputs;
	inputs[numberInputs++] = input;
}

unsigned Layer::getNumberInputs()
{
	return numberInputs;
}

Vector* Layer::getInput(unsigned pos)
{
	if (numberInputs == 0) {
		string error = "Trying to access an input of a Layer without inputs.";
		throw error;
	}
	if (pos > numberInputs){
		char buffer[100];
		sprintf(buffer, "Cannot access input %d: there are just %d inputs.", pos, numberInputs);
		string error = buffer;
		throw error;
	}

	return inputs[pos];
}

void Layer::calculateOutput()
{
	std::string error = "calculateOutput is not implemented.";
	throw error;
}

Vector* Layer::getOutput()
{
	return output;
}

void Layer::setSizes(unsigned totalWeighsPerOutput, unsigned outputSize)
{
	std::string error = "setSizes is not implemented.";
	throw error;
}

void Layer::resetSize()
{
	if (!output) {
		string error = "Cannot reset the size of a Layer without output.";
		throw error;
	}

	if (output != NULL){
		setSize(output->getSize());
	}
}

void Layer::setSize(unsigned size)
{
	unsigned auxTotalSize = 0;

	if (inputs != NULL){
		for (unsigned i=0; i < numberInputs; i++){
			auxTotalSize += getInput(i)->getWeighsSize();
		}
	}
	setSizes(auxTotalSize, size);
}

void Layer::saveWeighs(FILE* stream)
{
	std::string error = "saveWeighs is not implemented.";
	throw error;
}


void Layer::loadWeighs(FILE* stream)
{
	std::string error = "loadWeighs is not implemented.";
	throw error;
}

void Layer::save(FILE* stream)
{
	unsigned outputSize = output->getSize();

	fwrite(&inputType, sizeof(VectorType), 1, stream);
	fwrite(&outputType, sizeof(VectorType), 1, stream);
	fwrite(&functionType, sizeof(FunctionType), 1, stream);

	fwrite(&totalWeighsPerOutput, sizeof(unsigned), 1, stream);
	fwrite(&outputSize, sizeof(unsigned), 1, stream);

	saveWeighs(stream);
}

void Layer::load(FILE* stream)
{
	unsigned outputSize;

	fread(&inputType, sizeof(VectorType), 1, stream);
	fread(&outputType, sizeof(VectorType), 1, stream);
	fread(&functionType, sizeof(FunctionType), 1, stream);

	fread(&totalWeighsPerOutput, sizeof(unsigned), 1, stream);

	fread(&outputSize, sizeof(unsigned), 1, stream);
	setSizes(totalWeighsPerOutput, outputSize);

	loadWeighs(stream);

	inputs = NULL;
	numberInputs = 0;
}

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
}

void Layer::mutateWeigh(float mutationRange)
{
	unsigned chosenOutputOffset = randomUnsigned(this->output->getSize()) * this->totalWeighsPerOutput;
	unsigned chosenInput = randomUnsigned(this->numberInputs);
	unsigned chosenInputOffset = 0;
	for (unsigned i=0; i < chosenInput; i++) {
		chosenInputOffset += this->inputs[i]->getSize();
	}

	unsigned chosenWeigh = chosenOutputOffset + chosenInputOffset +
			randomUnsigned(this->inputs[chosenInput]->getSize());

	if (inputType == FLOAT){
		((float*) this->weighs)[chosenWeigh] += randomFloat(mutationRange);
	} else {
		unsigned discreteRange;
		if (mutationRange >= 128){
			discreteRange = 127;
		} else {
			discreteRange = (unsigned) mutationRange;
		}
		//TODO impedir que el unsigned char dé la vuelta?
		((unsigned char*) this->weighs)[chosenWeigh] += randomInt(discreteRange);
	}
}

void Layer::mutateWeighs(float probability, float mutationRange)
{
	if (!weighs) {
		string error = "Cannot mutate a Layer without weighs.";
		throw error;
	}
	unsigned discreteRange;
	if (inputType != FLOAT) {
		if (mutationRange >= 128){
			discreteRange = 127;
		} else {
			discreteRange = (unsigned) mutationRange;
		}
	}
	for (unsigned i=0; i < output->getSize(); i++){
		unsigned inputOffset = 0;
		for (unsigned j=0; j < numberInputs; j++){
			for (unsigned k=0; k < getInput(j)->getSize(); k++){
				unsigned weighPos = i*totalWeighsPerOutput + inputOffset + k;
				if (randomPositiveFloat(1) < probability) {
					if (inputType == FLOAT) {
						((float*)weighs)[weighPos] += randomFloat(mutationRange);
					} else {
						((unsigned char*)weighs)[weighPos] += randomInt(discreteRange);
					}
				}
			}
			inputOffset += getInput(j)->getWeighsSize();
		}
		if (randomPositiveFloat(1) < probability) {
			thresholds[i] += randomFloat(mutationRange);
		}
	}
}

Layer** Layer::crossoverWeighs(Layer *other, Interface *bitVector)
{
	if (bitVector->getSize() != this->getNumberWeighs()){
		string error = "The number of weighs must be equal to the size of the bitVector.";
		throw error;
	}
	if (!weighs) {
		string error = "Cannot crossover a Layer without weighs.";
		throw error;
	}
	Layer** twoLayers = (Layer**) mi_malloc(2 * sizeof(Layer*));
	twoLayers[0] = this->newCopy();
	twoLayers[1] = other->newCopy();

	unsigned vectorPos = 0;
	for (unsigned i=0; i < output->getSize(); i++){
		unsigned inputOffset = 0;

		if (bitVector->getElement(vectorPos++)) {
			twoLayers[0]->setThreshold(this->getThreshold(i), i);
			twoLayers[1]->setThreshold(other->getThreshold(i), i);
		} else {
			twoLayers[0]->setThreshold(other->getThreshold(i), i);
			twoLayers[1]->setThreshold(this->getThreshold(i), i);
		}

		for (unsigned j=0; j < numberInputs; j++){
			for (unsigned k=0; k < getInput(j)->getSize(); k++){
				unsigned weighPos = i*totalWeighsPerOutput + inputOffset + k;
				if (bitVector->getElement(vectorPos++)) {
					if (inputType == FLOAT) {
						twoLayers[0]->setFloatWeigh(this->getFloatWeigh(weighPos), weighPos);
						twoLayers[1]->setFloatWeigh(other->getFloatWeigh(weighPos), weighPos);
					} else {
						twoLayers[0]->setByteWeigh(this->getByteWeigh(weighPos), weighPos);
						twoLayers[1]->setByteWeigh(other->getByteWeigh(weighPos), weighPos);
					}
				} else {
					if (inputType == FLOAT) {
						twoLayers[0]->setFloatWeigh(other->getFloatWeigh(weighPos), weighPos);
						twoLayers[1]->setFloatWeigh(this->getFloatWeigh(weighPos), weighPos);
					} else {
						twoLayers[0]->setByteWeigh(other->getByteWeigh(weighPos), weighPos);
						twoLayers[1]->setByteWeigh(this->getByteWeigh(weighPos), weighPos);
					}
				}
			}
			inputOffset += getInput(j)->getWeighsSize();
		}
	}
	return twoLayers;
}

Layer** Layer::crossoverNeurons(Layer *other, Interface* bitVector)
{
	if (bitVector->getSize() != output->getSize()){
		string error = "The number of neurons must be equal to the size of the bitVector.";
		throw error;
	}
	if (!weighs) {
		string error = "Cannot crossover a Layer without weighs.";
		throw error;
	}
	Layer* offSpring = other->newCopy();
	Layer** twoLayers = (Layer**) mi_malloc(2 * sizeof(Layer*));
	twoLayers[0] = this->newCopy();
	twoLayers[1] = other->newCopy();

	size_t size;
	if (inputType == FLOAT){
		size = totalWeighsPerOutput * sizeof(float);
	} else {
		size = totalWeighsPerOutput * sizeof(unsigned char);
	}

	void* destination_a_ptr = twoLayers[0]->getWeighsPtr();
	void* destination_b_ptr = twoLayers[1]->getWeighsPtr();
	void* thisPtr = this->getWeighsPtr();
	void* otherPtr = other->getWeighsPtr();

	for (unsigned i=0; i < output->getSize(); i++){

		if (bitVector->getElement(i)) {
			memcpy(destination_a_ptr, thisPtr, size);
			twoLayers[0]->setThreshold(this->getThreshold(i), i);

			memcpy(destination_b_ptr, otherPtr, size);
			twoLayers[1]->setThreshold(other->getThreshold(i), i);
		} else {
			memcpy(destination_a_ptr, otherPtr, size);
			twoLayers[0]->setThreshold(other->getThreshold(i), i);

			memcpy(destination_b_ptr, thisPtr, size);
			twoLayers[1]->setThreshold(this->getThreshold(i), i);
		}
		destination_a_ptr = (void*) ((char*)destination_a_ptr + size);
		destination_b_ptr = (void*) ((char*)destination_b_ptr + size);
		thisPtr = (void*) ((char*)thisPtr + size);
		otherPtr = (void*) ((char*)otherPtr + size);
	}
	return twoLayers;
}

float Layer::getThreshold(unsigned  neuronPos)
{
	return thresholds[neuronPos];
}

void Layer::setThreshold(float value, unsigned  neuronPos)
{
	thresholds[neuronPos] = value;
}

unsigned char Layer::getByteWeigh(unsigned pos)
{
	return ((unsigned char*) this->weighs)[pos];
}

void Layer::setByteWeigh(unsigned char value, unsigned pos)
{
	((unsigned char*) this->weighs)[pos] = value;
}

float Layer::getFloatWeigh(unsigned pos)
{
	return ((float*) this->weighs)[pos];
}

void Layer::setFloatWeigh(float value, unsigned pos)
{
	((float*) this->weighs)[pos] = value;
}

void *Layer::getThresholdsPtr()
{
	return (void*)this->thresholds;
}

void *Layer::getWeighsPtr()
{
	return this->weighs;
}

unsigned Layer::getNumberNeurons()
{
	return this->output->getSize();
}

unsigned Layer::getNumberWeighs()
{
	unsigned numWeighs = 0;
	for (unsigned i=0; i < numberInputs; i++) {
		numWeighs += inputs[i]->getSize();
	}
	return this->output->getSize() * (numWeighs + 1);
}










