#include "cudaLayer.h"

unsigned CudaLayer::algorithm = 0;

CudaLayer::CudaLayer()
{
}

CudaLayer::CudaLayer(unsigned size, VectorType outputType, FunctionType functionType)
{
	init(size, outputType, functionType);
}

void CudaLayer::init(unsigned size, VectorType outputType, FunctionType functionType)
{
	output = new CudaVector(size, outputType, functionType);
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
		cuda_inputCalculation(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, Cuda_Threads_Per_Block);
	}
	else if (CudaLayer::algorithm == 1) {
		cuda_inputCalculation2(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, Cuda_Threads_Per_Block);
	}
}

float* CudaLayer::negativeThresholds()
{
	return cuda_getNegativeThresholds(thresholds, output->getSize(), Cuda_Threads_Per_Block);
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

	Vector* input = getInput(inputLayer);
	unsigned weighPos = (outputPos * input->getSize()) + inputPos;

	cuda_mutate(getWeighsPtr(inputLayer), weighPos, mutation, input->getVectorType());
}

void CudaLayer::mutateThreshold(unsigned outputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		string error = "Cannot mutate that Threshold: the Layer hasn't so many neurons.";
		throw error;
	}
	cuda_mutate(thresholds, outputPos, mutation, FLOAT);
}

void CudaLayer::crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector)
{
	unsigned weighsSize = bitVector->getSize();
	CudaVector* cudaBitVector = new CudaVector(weighsSize, BIT, Cuda_Threads_Per_Block);
	cudaBitVector->copyFrom2(bitVector, Cuda_Threads_Per_Block);
	unsigned* cudaBitVectorPtr = (unsigned*)cudaBitVector->getDataPointer();

	void* thisWeighs = this->getWeighsPtr(inputLayer);
	void* otherWeighs = other->getWeighsPtr(inputLayer);
	cuda_crossover(thisWeighs, otherWeighs, cudaBitVectorPtr, weighsSize, inputs[inputLayer]->getVectorType(), Cuda_Threads_Per_Block);
}





