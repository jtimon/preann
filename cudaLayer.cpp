#include "cudaLayer.h"

unsigned CudaLayer::algorithm = 0;
unsigned CudaLayer::blockSize = 128;

CudaLayer::CudaLayer(unsigned size, VectorType outputType, FunctionType functionType): Layer(outputType, functionType)
{
	output = new CudaVector(size, outputType);
	thresholds = (float*)cuda_malloc(sizeof(float) * size);
}

CudaLayer::~CudaLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			cuda_free(weighs[i]);
		}
		mi_free(inputs);
		mi_free(weighs);
	}
	if (output) {
		delete (output);
	}
	if (thresholds) {
		cuda_free(thresholds);
	}
}

void CudaLayer::inputCalculation(Vector* input, void* inputWeighs, float* results)
{
	if (CudaLayer::algorithm == 0) {
		cuda_inputCalculation(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, CudaLayer::blockSize);
	}
	else if (CudaLayer::algorithm == 1) {
		cuda_inputCalculation2(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, CudaLayer::blockSize);
	}
}

float* CudaLayer::negativeThresholds()
{
	return cuda_getNegativeThresholds(thresholds, output->getSize(), CudaLayer::blockSize);
}

void CudaLayer::saveWeighs(FILE *stream)
{
	unsigned size;

	size = output->getSize() * sizeof(float);
	float* aux_thresholds = (float*) mi_malloc(size);
	cuda_copyToHost(aux_thresholds, thresholds, size);
	fwrite(aux_thresholds, size, 1, stream);
	mi_free(aux_thresholds);

	for (unsigned i=0; i < numberInputs; i++){
		if (inputs[i]->getVectorType() == FLOAT){
			size = inputs[i]->getSize() * output->getSize() * sizeof(float);
		} else {
			size = inputs[i]->getSize() * output->getSize() * sizeof(unsigned char);
		}
		void* aux_weighs = mi_malloc(size);
		cuda_copyToHost(aux_weighs, weighs[i], size);
		fwrite(aux_weighs, size, 1, stream);
		mi_free(aux_weighs);
	}
}

void CudaLayer::loadWeighs(FILE *stream)
{
	unsigned size;

	size = output->getSize() * sizeof(float);
	float* aux_thresholds = (float*) mi_malloc(size);
	fread(aux_thresholds, size, 1, stream);
	cuda_copyToDevice(thresholds, aux_thresholds, size);
	mi_free(aux_thresholds);

	for (unsigned i=0; i < numberInputs; i++){
		if (inputs[i]->getVectorType() == FLOAT){
			size = inputs[i]->getSize() * output->getSize() * sizeof(float);
		} else {
			size = inputs[i]->getSize() * output->getSize() * sizeof(unsigned char);
		}

		void* aux_weighs = mi_malloc(size);
		fread(aux_weighs, size, 1, stream);
		cuda_copyToDevice(weighs[i], aux_weighs, size);
		mi_free(aux_weighs);
	}
}

void* CudaLayer::newWeighs(unsigned  inputSize, VectorType inputType)
{
	unsigned size;
	if (inputType == FLOAT) {
		size = output->getSize() * inputSize * sizeof(float);
	} else {
		size = output->getSize() * inputSize * sizeof(unsigned char);
	}
	return cuda_malloc(size);
}

void CudaLayer::copyWeighs(Layer* sourceLayer)
{
	//TODO implementar metodo
	std::string error = "newCopy is not implemented for CudaLayer.";
	throw error;
}

void CudaLayer::randomWeighs(float range)
{
	//TODO implementar metodo
	std::string error = "randomWeighs is not implemented for CudaLayer.";
	throw error;
}

void CudaLayer::mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation)
{
	//TODO implement method (pasar a cuda)
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

void CudaLayer::mutateThreshold(unsigned outputPos, float mutation)
{
	//TODO implement method (copiado de cpp) (pasar a cuda)
	if (outputPos > output->getSize()) {
		string error = "Cannot mutate that Threshold: the Layer hasn't so many neurons.";
		throw error;
	}
	thresholds[outputPos] += mutation;
}

void CudaLayer::crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector)
{
	//TODO implement method CudaLayer::crossoverWeighs
}





