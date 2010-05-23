
#include "cppLayer.h"

CppLayer::CppLayer(VectorType outputType, FunctionType functionType): Layer(outputType, functionType)
{
}

CppLayer::CppLayer(unsigned size, VectorType outputType, FunctionType functionType): Layer(outputType, functionType)
{
	output = new Vector(size, outputType);
	thresholds = (float*)mi_malloc(sizeof(float) * size);
}

CppLayer::~CppLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			mi_free(weighs[i]);
		}
		mi_free(inputs);
		mi_free(weighs);
		inputs = NULL;
		weighs = NULL;
	}
	if (thresholds) {
		mi_free(thresholds);
		thresholds = NULL;
	}
	if (output) {
		delete (output);
		output = NULL;
	}
}

float* CppLayer::negativeThresholds()
{
	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	for (unsigned j=0; j < output->getSize(); j++) {
		results[j] = -thresholds[j];
	}
	return results;
}

void CppLayer::inputCalculation(Vector* input, void* inputWeighs, float* results)
{
	void* inputPtr = input->getDataPointer();
	unsigned inputSize = input->getSize();

	for (unsigned j=0; j < output->getSize(); j++){

		for (unsigned k=0; k < inputSize; k++){

			unsigned weighPos = (j * inputSize) + k;
			if (input->getVectorType() == FLOAT) {
				//printf("i % d input %f weigh %f \n", k, ((float*)input)[k], ((float*)weighs)[weighPos]);
				results[j] += ((float*)inputPtr)[k] * ((float*)inputWeighs)[weighPos];
			} else {
				if ( ((unsigned*)inputPtr)[k/BITS_PER_UNSIGNED] & (0x80000000>>(k % BITS_PER_UNSIGNED)) ) {
					results[j] += (((unsigned char*)inputWeighs)[weighPos] - 128);
				} else if (input->getVectorType() == SIGN) {
					results[j] -= (((unsigned char*)inputWeighs)[weighPos] - 128);
				}
			}
		}
	}
}

void CppLayer::saveWeighs(FILE* stream)
{
	fwrite(thresholds, output->getSize() * sizeof(float), 1, stream);
	for (unsigned i=0; i < numberInputs; i++){
		unsigned size;
		if (inputs[i]->getVectorType() == FLOAT){
			size = output->getSize() * inputs[i]->getSize() * sizeof(float);
		} else {
			size = output->getSize() * inputs[i]->getSize() * sizeof(unsigned char);
		}
		fwrite(weighs[i], size, 1, stream);
	}
}

void CppLayer::loadWeighs(FILE* stream)
{
	fread(thresholds, output->getSize() * sizeof(float), 1, stream);
	for (unsigned i=0; i < numberInputs; i++){
		unsigned size;
		if (inputs[i]->getVectorType() == FLOAT){
			size = output->getSize() * inputs[i]->getSize() * sizeof(float);
		} else {
			size = output->getSize() * inputs[i]->getSize() * sizeof(unsigned char);
		}
		fread(weighs[i], size, 1, stream);
	}
}

void* CppLayer::newWeighs(unsigned inputSize, VectorType inputType)
{
	unsigned size;
	if (inputType == FLOAT) {
		size = output->getSize() * inputSize * sizeof(float);
	} else {
		size = output->getSize() * inputSize * sizeof(unsigned char);
	}
	return mi_malloc(size);
}

void CppLayer::copyWeighs(Layer* sourceLayer)
{
	memcpy(thresholds, sourceLayer->getThresholdsPtr(), output->getSize() * sizeof(float));

	for (unsigned i=0; i < numberInputs; i++){
		//TODO implementar metodo
	}
}

void CppLayer::randomWeighs(float range)
{
	if (output == NULL){
		string error = "Cannot set random weighs to a layer with no output.";
		throw error;
	}
	if (numberInputs == 0){
		string error = "Cannot set random weighs to a layer with no inputs.";
		throw error;
	}
	unsigned charRange;
	if (range >= 128){
		charRange = 127;
	} else {
		charRange = (unsigned)range;
	}
	for (unsigned i=0; i < output->getSize(); i++){

		//TODO esto peta con BIT inputType = BIT maxSize = 163840
		//thresholds[i] = 0;
		thresholds[i] = randomFloat(range);
		for (unsigned j=0; j < numberInputs; j++){

			unsigned inputSize = inputs[j]->getSize();
			for (unsigned k=0; k < inputSize; k++){

				unsigned weighPos = (i * inputSize) + k;
				if (inputs[j]->getVectorType() == FLOAT) {
					((float**)weighs)[j][weighPos] = randomFloat(range);
				} else {
					((unsigned char**)weighs)[j][weighPos] = 128 + (unsigned char)randomInt(charRange);
				}
			}
		}
	}
}

void CppLayer::mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		string error = "Cannot mutate that output: the Layer hasn't so many neurons.";
		throw error;
	}
	if (inputLayer > output->getSize()) {
		string error = "Cannot mutate that input: the Layer hasn't so many inputs.";
		throw error;
	}
	if (inputPos > inputs[inputLayer]->getSize()) {
		string error = "Cannot mutate that input: the input hasn't so many neurons.";
		throw error;
	}

	unsigned weighPos = (outputPos * inputs[inputLayer]->getSize()) + inputPos;

	if (inputs[inputLayer]->getVectorType() == FLOAT){
		((float**)weighs)[inputLayer][weighPos] += mutation;
	} else {

		int result = (int)mutation + ((unsigned char**)weighs)[inputLayer][weighPos];
		if (result <= 0){
			((unsigned char**)weighs)[inputLayer][weighPos] = 0;
		}
		else if (result >= 255) {
			((unsigned char**)weighs)[inputLayer][weighPos] = 255;
		}
		else {
			((unsigned char**)weighs)[inputLayer][weighPos] = result;
		}
	}
}

void CppLayer::mutateThreshold(unsigned outputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		string error = "Cannot mutate that Threshold: the Layer hasn't so many neurons.";
		throw error;
	}
	thresholds[outputPos] += mutation;
}

void CppLayer::crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector)
{
	//TODO CppLayer::crossoverWeighs
}


