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
	if (inputs) delete[] inputs;
	if (weighs) free(weighs);
	//TODO averiguar por qué cojones peta aquí
	//if (thresholds) delete[] thresholds;
	if (output) delete (output);
}

Vector* Layer::newVector(unsigned size, VectorType vectorType)
{
	return new Vector(size, vectorType);
}

void Layer::randomWeighs(float range)
{
	if (output == NULL){
		cout<<"Error: There's no output for this layer."<<endl;
	} else if (numberInputs == 0){
		cout<<"Error: This layer has no input."<<endl;
	} else {
		if (inputType != FLOAT && range >= 128){
			range = 127;
		}
		for (unsigned i=0; i < output->getSize(); i++){

			thresholds[i] = randomFloat(range);
			unsigned inputOffset = 0;
			for (unsigned j=0; j < numberInputs; j++){

				unsigned inputSize = inputs[j]->getSize();
				for (unsigned k=0; k < inputSize; k++){

					unsigned weighPos = i*totalWeighsPerOutput + inputOffset + k;
					if (inputType == FLOAT) {
						((float*)weighs)[weighPos] = randomFloat(range);
					} else {
						//TODO revisar el xmm a ver si se pueden usar char normales para los pesos (y no hacer el truco del 128)
						((unsigned char*)weighs)[weighPos] = 128 + (unsigned char)randomInt(range);
					}
				}
				inputOffset += inputs[j]->getWeighsSize();
			}
		}
	}
}

unsigned char Layer::addInput(Vector* input)
{
	if (input->getVectorType() == inputType) {

		if (numberInputs == 0){
			inputs = new Vector*[2];
		}
		if (numberInputs <= 1){
			inputs[numberInputs] = input;
		}
		else {
			Vector** newInputs = new Vector*[numberInputs + 1];
			memcpy(newInputs, inputs, numberInputs * sizeof(Vector*));
			newInputs[numberInputs] = input;
			delete [] inputs;
			inputs = newInputs;
		}
		numberInputs++;
		return 1;
	}
	cout<<"Error: unexpected input type."<<endl;
	return 0;
}

unsigned Layer::getNumberInputs()
{
	return numberInputs;
}

Vector* Layer::getInput(unsigned pos)
{
	if (pos > numberInputs){
		cout<<"Error: cannot access input "<<pos<<": there are just "<<numberInputs<<" inputs"<<endl;
		return NULL;
	} else {
		return inputs[pos];
	}
}

void Layer::calculateOutput()
{
	float result;
	for (unsigned i=0; i < output->getSize(); i++){
		result = 0;
		unsigned inputOffset = 0;
		for (unsigned j=0; j < numberInputs; j++){
			for (unsigned k=0; k < inputs[j]->getSize(); k++){
				unsigned weighPos = i*totalWeighsPerOutput + inputOffset + k;
				if (inputType == FLOAT) {
					result += inputs[j]->getElement(k) * ((float*)weighs)[weighPos];
				} else {
					result += inputs[j]->getElement(k) * (((unsigned char*)weighs)[weighPos] - 128);
				}
			}
			inputOffset += inputs[j]->getWeighsSize();
		}
		output->setElement(i, Function(result - thresholds[i], functionType));
	}
}

Vector* Layer::getOutput()
{
	return output;
}

void Layer::setSizes(unsigned totalWeighsPerOutput, unsigned ouputSize)
{
	if (output != NULL){
		delete (output);
		delete[] thresholds;
	}
	output = newVector(ouputSize, outputType);
	thresholds = new float[ouputSize];
	if (totalWeighsPerOutput > 0){
		if (inputType == FLOAT){
			weighs = new float[ouputSize * totalWeighsPerOutput];
			for (unsigned i=0; i < ouputSize * totalWeighsPerOutput; i++){
				((float*)weighs)[i] = 0;
			}
		} else {
			weighs = new unsigned char[ouputSize * totalWeighsPerOutput];
			for (unsigned i=0; i < ouputSize * totalWeighsPerOutput; i++){
				((unsigned char*)weighs)[i] = 128;
			}
		}
	}
	this->totalWeighsPerOutput = totalWeighsPerOutput;
}

void Layer::resetSize()
{
	if (output != NULL){
		setSize(output->getSize());
	}
}

void Layer::setSize(unsigned size)
{
	if (inputs == NULL){

		cout<<"Warning: cannot set the weighs of a Layer without inputs."<<endl;
	}
	unsigned auxTotalSize = 0;
	for (unsigned i=0; i < numberInputs; i++){
		auxTotalSize += inputs[i]->getWeighsSize();
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
