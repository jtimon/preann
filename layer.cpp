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
	//TODO poner aqui lo que hay en freeLayer() y evitar que pete
}

void Layer::freeLayer()
{
	if (inputs) {
		free(inputs);
	}
	if (weighs) {
		free(weighs);
	}
	if (thresholds) {
		free(thresholds);
	}
	if (output) {
		output->freeVector();
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

	Vector** newInputs = (Vector**) malloc(sizeof(Vector*) * (numberInputs + 1));
	memcpy(newInputs, inputs, numberInputs * sizeof(Vector*));
	newInputs[numberInputs] = input;
	if (inputs) {
		free(inputs);
	}
	inputs = newInputs;
	numberInputs++;
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
				}
			}
			inputOffset += getInput(j)->getWeighsSize();
		}
		output->setElement(i, Function(result - thresholds[i], functionType));
	}
}

Vector* Layer::getOutput()
{
	return output;
}

void Layer::setSizes(unsigned totalWeighsPerOutput, unsigned outputSize)
{
	if (!output) {
		output = newVector(outputSize, outputType);
		thresholds = (float*) malloc(sizeof(float) * outputSize);
	} else if (output->getSize() != outputSize) {

		cout<<"Warning: a layer is changing the location of its output."<<endl;
		delete (output);
		if (thresholds) {
			free(thresholds);
		}
		output = newVector(outputSize, outputType);
		thresholds = (float*)malloc(sizeof(float) * outputSize);
	}
	if (totalWeighsPerOutput > 0){
		if (inputType == FLOAT){

			weighs = new float[outputSize * totalWeighsPerOutput];
			for (unsigned i=0; i < outputSize * totalWeighsPerOutput; i++){
				((float*)weighs)[i] = 0;
			}
		} else {
			weighs = new unsigned char[outputSize * totalWeighsPerOutput];
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
