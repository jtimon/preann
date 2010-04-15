#include "layer.h"

Layer::Layer(VectorType outputType, FunctionType functionType)
{
	inputs = NULL;
	weighs = NULL;
	numberInputs = 0;

	thresholds = NULL;
	output = NULL;

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

void Layer::checkCompatibility(Layer* layer)
{
	if (this->getImplementationType() != layer->getImplementationType()){
		string error = "The layers are incompatible: the implementation is different.";
		throw error;
	}
	if (this->getOutput()->getSize() != layer->getOutput()->getSize()){
		string error = "The layers are incompatible: the output size is different.";
		throw error;
	}
	if (this->getOutput()->getVectorType() != layer->getOutput()->getVectorType()){
		string error = "The layers are incompatible: the output type is different.";
		throw error;
	}
	if (this->getNumberInputs() != layer->getNumberInputs()){
		string error = "The layers are incompatible: the number of inputs is different.";
		throw error;
	}
	for (unsigned i=0; i < numberInputs; i++){
		if (this->getInput(i)->getSize() != layer->getInput(i)->getSize()){
			string error = "The layers are incompatible: the size of an input is different.";
			throw error;
		}
		if (this->getInput(i)->getVectorType() != layer->getInput(i)->getVectorType()){
			string error = "The layers are incompatible: the type of an input is different.";
			throw error;
		}
	}

}

void Layer::calculateOutput()
{
	if (!output) {
		string error = "Cannot calculate the output of a Layer without output.";
		throw error;
	}

	float* results = negativeThresholds();

	for(unsigned i=0; i < numberInputs; i++){
		inputCalculation(inputs[i], weighs[i], results);
	}
//	printf("----------------\n", 1);
//	for (unsigned i=0; i < output->getSize(); i++){
//		printf("%f ", results[i]);
//	}
//	printf("\n----------------\n", 1);
	output->activation(results, functionType);
	output->print();
}

void Layer::addInput(Vector* input)
{
	//TODO probar que sucede con varios tipos de entrada
	Vector** newInputs = (Vector**) mi_malloc(sizeof(Vector*) * (numberInputs + 1));
	void** newWeighsPtr = (void**) mi_malloc(sizeof(void*) * (numberInputs + 1));
	if (inputs) {
		memcpy(newInputs, inputs, numberInputs * sizeof(Vector*));
		memcpy(newWeighsPtr, weighs, numberInputs * sizeof(void*));
		mi_free(inputs);
	}
	inputs = newInputs;
	weighs = newWeighsPtr;

	inputs[numberInputs] = input;
	newWeighsPtr[numberInputs] = newWeighs(input->getSize(), input->getVectorType());
	++numberInputs;
}

void Layer::save(FILE* stream)
{
	fwrite(&functionType, sizeof(FunctionType), 1, stream);

	saveWeighs(stream);
}

void Layer::load(FILE* stream)
{
	fread(&functionType, sizeof(FunctionType), 1, stream);

	loadWeighs(stream);
}

void Layer::swapWeighs(Layer* layer)
{
	checkCompatibility(layer);

	unsigned size = numberInputs * sizeof(void*);
	void** temp = (void**) mi_malloc(size);
	memcpy(temp, weighs, size);
	memcpy(weighs, layer->weighs, size);
	memcpy(layer->weighs, temp, size);
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
	return thresholds;
}

void *Layer::getWeighsPtr()
{
	return weighs;
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
	Interface* inputBitVector = new Interface(inputSize, BIT);

	for (unsigned i=0; i < inputSize; i++){
		if (bitVector->getElement(i)){
			for (unsigned j=0; j < output->getSize(); j++){
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
	//TODO comprobar todo esto en un nivel superior (o no comprobarlo) (o comprobarlo optionalmente)
	checkCompatibility(other);

	for (unsigned i=0; i < numberInputs; i++){

		unsigned inputSize = inputs[i]->getSize();
		Interface* inputBitVector = new Interface(inputSize, BIT);

		for (unsigned j=0; j < output->getSize(); j++){
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
