/*
 * CudaVector2.cpp
 *
 *  Created on: Oct 29, 2010
 *      Author: timon
 */

#include "cudaVector2.h"

CudaVector2::CudaVector2(unsigned size, VectorType vectorType, unsigned block_size)
{
	(((size-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	this->size = size;
	this->vectorType = vectorType;

	unsigned byte_sz = ((size-1)/(BITS_PER_UNSIGNED * block_size)+1) * (sizeof(unsigned) * block_size);
	data = cuda_malloc(byte_sz);

	cuda_setZero(data, byte_sz, vectorType, CUDA_THREADS_PER_BLOCK);
}

CudaVector2::CudaVector2(unsigned size, VectorType vectorType)
{
	this->size = size;
	this->vectorType = vectorType;

	unsigned byte_sz = getByteSize();
	data = cuda_malloc(byte_sz);

	cuda_setZero(data, byte_sz, vectorType, CUDA_THREADS_PER_BLOCK);
}

CudaVector2::~CudaVector2()
{
	if (data) {
		cuda_free(data);
		data = NULL;
	}
}

Vector* CudaVector2::clone()
{
	//TODO implementar CudaVector2::clone()
	Vector* clone = new CudaVector2(size, vectorType);
	copyToVector(clone);
	return clone;
}

void CudaVector2::inputCalculation(Vector* input, Vector* inputWeighsVect)
{
	void* inputWeighs = inputWeighsVect->getDataPointer();
	float* results = (float*)this->getDataPointer();

	cuda_inputCalculationInvertedMatrix(input->getDataPointer(), input->getSize(), input->getVectorType(), size, inputWeighs, results, Cuda_Threads_Per_Block);
}

void CudaVector2::mutate(unsigned pos, float mutation, unsigned inputSize)
{
	if (pos > size){
		std::string error = "The position being mutated is greater than the size of the vector.";
		throw error;
	}
	//TODO simplificar cuentas
	unsigned outputPos = pos / inputSize;
	unsigned inputPos = (pos % inputSize);
	unsigned outputSize = size / inputSize;
	unsigned weighPos = outputPos + (inputPos * outputSize);
	cuda_mutate(data, weighPos, mutation, vectorType);
}
void CudaVector2::weighCrossover(Vector* other, Interface* bitVector, unsigned inputSize)
{
	//TODO impl CudaVector2::weighCrossover
    if(size != other->getSize()){
        std::string error = "The vectors must have the same size to crossover them.";
        throw error;
    }
    if(vectorType != other->getVectorType()){
        std::string error = "The vectors must have the same type to crossover them.";
        throw error;
    }

	Interface invertedBitVector = Interface(size, BIT);

	unsigned width = size / inputSize;
	unsigned height = inputSize;

	for (unsigned i=0; i < width; i++){
		for (unsigned j=0; j < height; j++){
			invertedBitVector.setElement(i  + (j * width), bitVector->getElement((i * height) + j));
		}
	}

    CudaVector2* cudaBitVector = new CudaVector2(size, BIT, Cuda_Threads_Per_Block);
    cudaBitVector->copyFrom2(&invertedBitVector, Cuda_Threads_Per_Block);

    unsigned* cudaBitVectorPtr = (unsigned*)(cudaBitVector->getDataPointer());

    cuda_crossover(this->getDataPointer(), other->getDataPointer(), cudaBitVectorPtr, size, vectorType, Cuda_Threads_Per_Block);

    delete(cudaBitVector);
}
