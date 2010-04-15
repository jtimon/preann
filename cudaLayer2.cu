/*
 * cudaLayer2.cu
 *
 *  Created on: Apr 15, 2010
 *      Author: timon
 */

#include "cudaLayer2.h"

CudaLayer2::CudaLayer2(unsigned size, VectorType outputType, FunctionType functionType) : CudaLayer(size, outputType, functionType)
{
}

CudaLayer2::~CudaLayer2()
{
}

void* CudaLayer2::weighsToNormalPosition(void *sourceWeighs, unsigned numInput)
{
	Vector* input = inputs[numInput];

	unsigned size;
	if (input->getVectorType() == FLOAT){
		size = input->getSize() * output->getSize() * sizeof(float);
	} else {
		size = input->getSize() * output->getSize() * sizeof(unsigned char);
	}
	void* destinationWeighs = mi_malloc(size);

	for (unsigned j=0; j < output->getSize(); j++){
		for (unsigned k=0; k < input->getSize(); k++){

			unsigned weighSourcePos = j + (k * output->getSize());
			unsigned weighDestinationPos = (j * input->getSize()) + k;

			if (input->getVectorType() == FLOAT) {
				((float*)sourceWeighs)[weighSourcePos] = ((float*)destinationWeighs)[weighDestinationPos];
			} else {
				((unsigned char*)sourceWeighs)[weighSourcePos] = ((unsigned char*)destinationWeighs)[weighDestinationPos];
			}
		}
	}

	mi_free(sourceWeighs);
	return destinationWeighs;
}

void* CudaLayer2::weighsToCudaPosition(void *sourceWeighs, unsigned numInput)
{
	Vector* input = inputs[numInput];

	unsigned size;
	if (input->getVectorType() == FLOAT){
		size = input->getSize() * output->getSize() * sizeof(float);
	} else {
		size = input->getSize() * output->getSize() * sizeof(unsigned char);
	}
	void* destinationWeighs = mi_malloc(size);

	for (unsigned j=0; j < output->getSize(); j++){
		for (unsigned k=0; k < input->getSize(); k++){

			unsigned weighSourcePos = (j * input->getSize()) + k;
			unsigned weighDestinationPos = j + (k * output->getSize());

			if (input->getVectorType() == FLOAT) {
				((float*)sourceWeighs)[weighSourcePos] = ((float*)destinationWeighs)[weighDestinationPos];
			} else {
				((unsigned char*)sourceWeighs)[weighSourcePos] = ((unsigned char*)destinationWeighs)[weighDestinationPos];
			}
		}
	}

	mi_free(sourceWeighs);
	return destinationWeighs;
}

void CudaLayer2::crossoverWeighs(Layer *other, unsigned  inputLayer, Interface *bitVector)
{
	// TODO CudaLayer2::crossoverWeighs
}

void CudaLayer2::mutateWeigh(unsigned  outputPos, unsigned  inputLayer, unsigned  inputPos, float mutation)
{
	// TODO CudaLayer2::mutateWeigh
}

void CudaLayer2::inputCalculation(Vector *input, void *inputWeighs, float *results)
{
	cuda_inputCalculation4(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, CudaLayer::blockSize);
}

void CudaLayer2::loadWeighs(FILE *stream)
{
	unsigned size;

	size = output->getSize() * sizeof(float);
	float* aux_thresholds = (float*) mi_malloc(size);
	fread(aux_thresholds, size, 1, stream);
	cudaMemcpy(thresholds, thresholds, size, cudaMemcpyHostToDevice);
	mi_free(aux_thresholds);

	for (unsigned i=0; i < numberInputs; i++){
		if (inputs[i]->getVectorType() == FLOAT){
			size = inputs[i]->getSize() * output->getSize() * sizeof(float);
		} else {
			size = inputs[i]->getSize() * output->getSize() * sizeof(unsigned char);
		}

		void* aux_weighs = mi_malloc(size);
		fread(aux_weighs, size, 1, stream);
		aux_weighs = weighsToCudaPosition(aux_weighs, i);
		cudaMemcpy(weighs[i], aux_weighs, size, cudaMemcpyHostToDevice);
		mi_free(aux_weighs);
	}
	checkCUDAError("CudaLayer2::loadWeighs");
}

void CudaLayer2::saveWeighs(FILE *stream)
{
	unsigned size;

	size = output->getSize() * sizeof(float);
	float* aux_thresholds = (float*) mi_malloc(size);
	cudaMemcpy(aux_thresholds, thresholds, size, cudaMemcpyDeviceToHost);
	fwrite(aux_thresholds, size, 1, stream);
	mi_free(aux_thresholds);

	for (unsigned i=0; i < numberInputs; i++){
		if (inputs[i]->getVectorType() == FLOAT){
			size = inputs[i]->getSize() * output->getSize() * sizeof(float);
		} else {
			size = inputs[i]->getSize() * output->getSize() * sizeof(unsigned char);
		}
		void* aux_weighs = mi_malloc(size);
		cudaMemcpy(aux_weighs, weighs[i], size, cudaMemcpyDeviceToHost);
		aux_weighs = weighsToNormalPosition(aux_weighs, i);
		fwrite(aux_weighs, size, 1, stream);
		mi_free(aux_weighs);
	}
	checkCUDAError("CudaLayer2::saveWeighs");
}
