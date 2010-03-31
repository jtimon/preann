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

float* CudaLayer::negativeThresholds()
{
	return cuda_getNegativeThresholds(thresholds, output->getSize(), CudaLayer::blockSize);
}

void CudaLayer::inputCalculation(Vector* input, void* inputWeighs, float* results)
{
	if (CudaLayer::algorithm == 0){
		cuda_inputCalculation(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, CudaLayer::blockSize);
	} else {
		cuda_inputCalculation2(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, CudaLayer::blockSize);
	}
}

void CudaLayer::randomWeighs(float range)
{
	std::string error = "randomWeighs is not implemented for CudaLayer.";
	throw error;
}
/*
void CudaLayer::calculateOutput()
{
	float* results =

	for(unsigned i=0; i < numberInputs; i++){
		Vector* input = inputs[i];
		if (CudaLayer::algorithm == 0){
			cuda_inputCalculation(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), weighs[i], results, CudaLayer::blockSize);
		} else {
			cuda_inputCalculation2(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), weighs[i], results, CudaLayer::blockSize);
		}
	}
//	printf("----------------\n", 1);
//	for (unsigned i=0; i < output->getSize(); i++){
//		printf("%f ", results[i]);
//	}
//	printf("\n----------------\n", 1);
	output->activation(results, functionType);
}
*/
Layer* CudaLayer::newCopy()
{
	std::string error = "newCopy is not implemented for CudaLayer.";
	throw error;
}

