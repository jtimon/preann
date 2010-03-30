/*
 * cudaLayer2.cpp
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#include "cudaLayer.h"

unsigned CudaLayer::algorithm = 0;
unsigned CudaLayer::blockSize = 128;

CudaLayer::CudaLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
	// TODO Auto-generated constructor stub
}

CudaLayer::~CudaLayer()
{
	if (inputs) {
		mi_free(inputs);
	}
	if (output) {
		delete (output);
	}

	if (thresholds) {
		cuda_free(thresholds);
	}
	if (weighs) {
		cuda_free(weighs);
	}
}

void CudaLayer::setSizes(unsigned  totalWeighsPerOutput, unsigned  outputSize)
{
	if (!output) {
		output = new CudaVector(outputSize, outputType);
		thresholds = (float*) cuda_malloc(sizeof(float) * outputSize);
	} else if (output->getSize() != outputSize) {

		cout<<"Warning: a layer is changing the location of its output."<<endl;
		delete (output);
		if (thresholds) {
			cuda_free(thresholds);
		}
		output = new CudaVector(outputSize, outputType);
		thresholds = (float*)cuda_malloc(sizeof(float) * outputSize);
	}
	if (totalWeighsPerOutput > 0){
		unsigned weighs_size = outputSize * totalWeighsPerOutput;
		if (inputType == FLOAT){
			weighs = cuda_malloc(sizeof(float) * weighs_size);
		} else {
			weighs = cuda_malloc(sizeof(unsigned char) * weighs_size);
		}
	}
	this->totalWeighsPerOutput = totalWeighsPerOutput;
}

Layer* CudaLayer::newCopy()
{
	std::string error = "save is not implemented for newCopy.";
	throw error;
}

void CudaLayer::saveWeighs(FILE *stream)
{
	unsigned size;

	size = output->getSize() * sizeof(float);
	float* aux_thresholds = (float*) mi_malloc(size);
	cuda_copyToHost(aux_thresholds, thresholds, size);
	fwrite(aux_thresholds, size, 1, stream);
	mi_free(aux_thresholds);

	if (inputType == FLOAT){
		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
	} else {
		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
	}
	void* aux_weighs = mi_malloc(size);
	cuda_copyToHost(aux_weighs, weighs, size);
	fwrite(aux_weighs, size, 1, stream);
	mi_free(aux_weighs);
}

void CudaLayer::loadWeighs(FILE *stream)
{
	unsigned size;

	size = output->getSize() * sizeof(float);
	float* aux_thresholds = (float*) mi_malloc(size);
	fread(aux_thresholds, size, 1, stream);
	cuda_copyToDevice(thresholds, aux_thresholds, size);
	mi_free(aux_thresholds);

	if (inputType == FLOAT){
		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
	} else {
		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
	}
	void* aux_weighs = mi_malloc(size);
	fread(aux_weighs, size, 1, stream);
	cuda_copyToDevice(weighs, aux_weighs, size);
	mi_free(aux_weighs);
}

void CudaLayer::randomWeighs(float range)
{
	std::string error = "randomWeighs is not implemented for CudaLayer2.";
	throw error;
}

Vector* CudaLayer::newVector(unsigned  size, VectorType vectorType)
{
	return new CudaVector(size, vectorType);
}

void CudaLayer::calculateOutput()
{
	float* results = cuda_getNegativeThresholds(thresholds, output->getSize(), THREADS_PER_BLOCK);

	unsigned inputOffset = 0;
	for(unsigned i=0; i < numberInputs; i++){
		Vector* input = inputs[i];
		if (CudaLayer::algorithm == 0){
			cuda_inputCalculation(input->getDataPointer(), input->getSize(), input->getVectorType(), inputOffset, output->getSize(), weighs, totalWeighsPerOutput, results, CudaLayer::blockSize);
		} else {
			cuda_inputCalculation2(input->getDataPointer(), input->getSize(), input->getVectorType(), inputOffset, output->getSize(), weighs, totalWeighsPerOutput, results, CudaLayer::blockSize);
		}
		inputOffset += input->getSize();
	}
//	printf("----------------\n", 1);
//	for (unsigned i=0; i < output->getSize(); i++){
//		printf("%f ", results[i]);
//	}
//	printf("\n----------------\n", 1);
	output->activation(results, functionType);
}


