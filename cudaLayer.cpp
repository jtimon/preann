#include "cudaLayer.h"

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





