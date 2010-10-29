/*
 * cudaLayer2.cu
 *
 *  Created on: Apr 15, 2010
 *      Author: timon
 */

#include "cudaLayer2.h"

void CudaLayer2::crossoverWeighs(Layer *other, unsigned  inputLayer, Interface *bitVector)
{
	// TODO CudaLayer2::crossoverWeighs
	unsigned weighsSize = bitVector->getSize();
	Interface invertedBitVector = Interface(weighsSize, BIT);

	unsigned width = output->getSize();
	unsigned height = inputs[inputLayer]->getSize();

	for (unsigned i=0; i < width; i++){
		for (unsigned j=0; j < height; j++){
			invertedBitVector.setElement(i  + (j * width), bitVector->getElement((i * height) + j));
		}
	}

	CudaVector cudaBitVector = CudaVector(weighsSize, BIT, Cuda_Threads_Per_Block);
	cudaBitVector.copyFrom2(bitVector, Cuda_Threads_Per_Block);
	unsigned* cudaBitVectorPtr = (unsigned*)cudaBitVector.getDataPointer();

	void* thisWeighs = this->getConnection(inputLayer)->getDataPointer();
	void* otherWeighs = other->getConnection(inputLayer)->getDataPointer();
	cuda_crossover(thisWeighs, otherWeighs, cudaBitVectorPtr, weighsSize, inputs[inputLayer]->getVectorType(), Cuda_Threads_Per_Block);
}

