
#include "layer.h"

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
}

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
	if (inputs) {
		mi_free(inputs);
	}
	if (thresholds) {
		mi_free(thresholds);
	}
	if (weighs) {
		mi_free(weighs);
	}
	if (output) {
		delete (output);
	}
}

Vector* Layer::newVector(unsigned size, VectorType vectorType)
{
	return new Vector(size, vectorType);
}

void Layer::randomWeighs(float range)
{
	if (output == NULL){
		string error = "Cannot set random weighs to a layer with no output.";
		throw error;
	}
	if (numberInputs == 0){
		string error = "Cannot set random weighs to a layer with no inputs.";
		throw error;
	}
	if (inputType != FLOAT && range >= 128){
		range = 127;
	}
	for (unsigned i=0; i < output->getSize(); i++){

		thresholds[i] = randomFloat(range);
		unsigned inputOffset = 0;
		for (unsigned j=0; j < numberInputs; j++){

			unsigned inputSize = getInput(j)->getSize();
			for (unsigned k=0; k < inputSize; k++){

				unsigned weighPos = i*totalWeighsPerOutput + inputOffset + k;
				if (inputType == FLOAT) {
					((float*)weighs)[weighPos] = randomFloat(range);
				} else {
					//TODO revisar el xmm a ver si se pueden usar char normales para los pesos (y no hacer el truco del 128)
					((unsigned char*)weighs)[weighPos] = 128 + (unsigned char)randomInt(range);
				}
			}
			inputOffset += getInput(j)->getWeighsSize();
		}
	}
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
	if (!output) {
		string error = "Cannot calculate the output of a Layer without output.";
		throw error;
	}
	//printf("\n ", 1);
	float result;
	for (unsigned i=0; i < output->getSize(); i++){
		result = 0;
		unsigned inputOffset = 0;
		for (unsigned j=0; j < numberInputs; j++){
			for (unsigned k=0; k < getInput(j)->getSize(); k++){
				unsigned weighPos = i*totalWeighsPerOutput + inputOffset + k;
				if (inputType == FLOAT) {
					result += getInput(j)->getElement(k) * ((float*)weighs)[weighPos];
				} else {
					result += getInput(j)->getElement(k) * (((unsigned char*)weighs)[weighPos] - 128);
					//printf(" %d", (((unsigned char*)weighs)[weighPos] - 128));
					if (getInput(j)->getElement(k)) {
						//printf("X", 1);
					}
					//printf(" ", 1);
				}
			}
			inputOffset += getInput(j)->getWeighsSize();
		}
		printf(" %f ", result - thresholds[i]);
		output->setElement(i, Function(result - thresholds[i], functionType));
	}
	printf("\n ", 1);
}

Vector* Layer::getOutput()
{
	return output;
}

void Layer::setSizes(unsigned totalWeighsPerOutput, unsigned outputSize)
{
	if (!output) {
		output = newVector(outputSize, outputType);
		thresholds = (float*) mi_malloc(sizeof(float) * outputSize);
	} else if (output->getSize() != outputSize) {

		cout<<"Warning: a layer is changing the location of its output."<<endl;
		delete (output);
		if (thresholds) {
			mi_free(thresholds);
		}
		output = newVector(outputSize, outputType);
		thresholds = (float*)mi_malloc(sizeof(float) * outputSize);
	}
	if (totalWeighsPerOutput > 0){
		if (inputType == FLOAT){

			weighs = mi_malloc(sizeof(float) * outputSize * totalWeighsPerOutput);
			for (unsigned i=0; i < outputSize * totalWeighsPerOutput; i++){
				((float*)weighs)[i] = 0;
			}
		} else {
			weighs = mi_malloc(sizeof(unsigned char) * outputSize * totalWeighsPerOutput);
			for (unsigned i=0; i < outputSize * totalWeighsPerOutput; i++){
				((unsigned char*)weighs)[i] = 128;
			}
		}
	}
	this->totalWeighsPerOutput = totalWeighsPerOutput;
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

void Layer::save(FILE* stream)
{
	unsigned outputSize = output->getSize();

	fwrite(&inputType, sizeof(VectorType), 1, stream);
	fwrite(&outputType, sizeof(VectorType), 1, stream);
	fwrite(&functionType, sizeof(FunctionType), 1, stream);

	fwrite(&totalWeighsPerOutput, sizeof(unsigned), 1, stream);
	fwrite(&outputSize, sizeof(unsigned), 1, stream);

	fwrite(thresholds, outputSize * sizeof(float), 1, stream);
	unsigned size;
	if (inputType == FLOAT){
		size = outputSize * totalWeighsPerOutput * sizeof(float);
	} else {
		size = outputSize * totalWeighsPerOutput * sizeof(unsigned char);
	}
	fwrite(weighs, size, 1, stream);
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

	fread(thresholds, outputSize * sizeof(float), 1, stream);
	unsigned size;
	if (inputType == FLOAT){
		size = outputSize * totalWeighsPerOutput * sizeof(float);
	} else {
		size = outputSize * totalWeighsPerOutput * sizeof(unsigned char);
	}
	fread(weighs, size, 1, stream);

	inputs = NULL;
	numberInputs = 0;
}

Layer* Layer::newCopy()
{
	Layer* copy = new Layer(inputType, outputType, functionType);

	copy->setSizes(totalWeighsPerOutput, output->getSize());

	return copy;
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

Layer** Layer::crossoverWeighs(Layer *other, Vector *bitVector)
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

Layer** Layer::crossoverNeurons(Layer *other, Vector *bitVector)
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










