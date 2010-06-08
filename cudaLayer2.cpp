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

void CudaLayer2::transposeMatrix(void* matrix, unsigned width, unsigned height, VectorType inputType)
{
	unsigned size;
	void* auxMatrix;

	if (inputType == FLOAT){
		size = width * height * sizeof(float);

		auxMatrix = mi_malloc(size);

		for (unsigned i=0; i < width; i++){
			for (unsigned j=0; j < height; j++){
				((float*)auxMatrix)[(i * height) + j] = ((float*)matrix)[i + (j * width)];
			}
		}
	} else {
		size = width * height * sizeof(unsigned char);

		auxMatrix = mi_malloc(size);

		for (unsigned i=0; i < width; i++){
			for (unsigned j=0; j < height; j++){
				((unsigned char*)auxMatrix)[(i * height) + j] = ((unsigned char*)matrix)[i + (j * width)];
			}
		}
	}

	memcpy(matrix, auxMatrix, size);
	mi_free(auxMatrix);
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
	cuda_inputCalculation3(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, CudaLayer::blockSize);
}

void CudaLayer2::saveWeighs(FILE *stream)
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
		transposeMatrix(aux_weighs, output->getSize(), inputs[i]->getSize(), inputs[i]->getVectorType());
		fwrite(aux_weighs, size, 1, stream);
		mi_free(aux_weighs);
	}
}

void CudaLayer2::loadWeighs(FILE *stream)
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
		transposeMatrix(aux_weighs, inputs[i]->getSize(), output->getSize(), inputs[i]->getVectorType());
		cuda_copyToDevice(weighs[i], aux_weighs, size);
		mi_free(aux_weighs);
	}
}
