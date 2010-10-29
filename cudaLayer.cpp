#include "cudaLayer.h"

CudaLayer::CudaLayer()
{
}

CudaLayer::~CudaLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			delete(connections[i]);
		}
		mi_free(inputs);
		mi_free(connections);
	}
	if (output) {
		delete(output);
	}
	if (thresholds) {
		delete(thresholds);
	}
}

void CudaLayer::inputCalculation(Vector* input, Vector* inputWeighsVect, Vector* resultsVect)
{
	void* inputWeighs = inputWeighsVect->getDataPointer();
	float* results = (float*)resultsVect->getDataPointer();
	//FIXME este método no funciona correctamente para SIGN
	if (CudaVector::algorithm == 0) {
		cuda_inputCalculationReduction(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, Cuda_Threads_Per_Block);
	}
	else if (CudaVector::algorithm == 1) {
		cuda_inputCalculation(input->getDataPointer(), input->getSize(), input->getVectorType(), output->getSize(), inputWeighs, results, Cuda_Threads_Per_Block);
	}
}

float* CudaLayer::negativeThresholds()
{
	return cuda_getNegativeThresholds((float*)thresholds->getDataPointer(), output->getSize(), Cuda_Threads_Per_Block);
}

void CudaLayer::copyWeighs(Layer* sourceLayer)
{
	//TODO implementar metodo
	std::string error = "CudaLayer::copyWeighs is not implemented.";
	throw error;
}

void CudaLayer::randomWeighs(float range)
{
	//TODO implementar metodo
	std::string error = "CudaLayer::randomWeighs is not implemented.";
	throw error;
}

void CudaLayer::mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		std::string error = "Cannot mutate that output: the Layer hasn't so many neurons.";
		throw error;
	}
	if (inputLayer > output->getSize()) {
		std::string error = "Cannot mutate that input: the Layer hasn't so many inputs.";
		throw error;
	}
	if (inputPos > inputs[inputLayer]->getSize()) {
		std::string error = "Cannot mutate that input: the input hasn't so many neurons.";
		throw error;
	}

	Vector* input = getInput(inputLayer);
	unsigned weighPos = (outputPos * input->getSize()) + inputPos;

	cuda_mutate(getConnection(inputLayer)->getDataPointer(), weighPos, mutation, input->getVectorType());
}

void CudaLayer::mutateThreshold(unsigned outputPos, float mutation)
{
	if (outputPos > output->getSize()) {
		std::string error = "Cannot mutate that Threshold: the Layer hasn't so many neurons.";
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

	void* thisWeighs = this->getConnection(inputLayer)->getDataPointer();
	void* otherWeighs = other->getConnection(inputLayer)->getDataPointer();
	cuda_crossover(thisWeighs, otherWeighs, cudaBitVectorPtr, weighsSize, inputs[inputLayer]->getVectorType(), Cuda_Threads_Per_Block);
}





