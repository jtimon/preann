
#ifndef CUDAVECTOR_H_
#define CUDAVECTOR_H_

#include "vectorImpl.h"
#include "cuda_code.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class CudaVector: virtual public Vector, virtual public VectorImpl<vectorTypeTempl, c_typeTempl> {
protected:
	unsigned getByteSize();
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);
	CudaVector(Interface* bitVector, unsigned block_size);
public:
	CudaVector() {};
	CudaVector(unsigned size);
	virtual ~CudaVector();
	virtual ImplementationType getImplementationType() {
		return CUDA;
	};

	virtual Vector* clone();
	virtual void activation(Vector* results, FunctionType functionType);
};

template <VectorType vectorTypeTempl, class c_typeTempl>
CudaVector<vectorTypeTempl, c_typeTempl>::CudaVector(unsigned size)
{
	this->tSize = size;

	unsigned byte_sz = getByteSize();
	data = cuda_malloc(byte_sz);

	cuda_setZero(data, byte_sz, vectorTypeTempl, CUDA_THREADS_PER_BLOCK);
}

//special constructor for bit coalescing vectors
template <VectorType vectorTypeTempl, class c_typeTempl>
CudaVector<vectorTypeTempl, c_typeTempl>::CudaVector(Interface* bitVector, unsigned block_size)
{
	if (bitVector->getVectorType() != BIT){
		std::string error = "The Vector type must be BIT to use a BitVector CudaVector constructor.";
		throw error;
	}
	unsigned bitVectorSize = bitVector->getSize();
	unsigned maxWeighsPerBlock = BITS_PER_UNSIGNED * block_size;

	tSize = (bitVectorSize / maxWeighsPerBlock) * maxWeighsPerBlock;
	tSize += min(bitVectorSize % maxWeighsPerBlock, block_size) * BITS_PER_UNSIGNED;

	Interface interfaceOrderedByBlockSize = Interface(tSize, BIT);
	unsigned byteSize = interfaceOrderedByBlockSize.getByteSize();
	data = cuda_malloc(byteSize);

	unsigned bit = 0, thread = 0, block_offset = 0;
	for (unsigned i=0; i < bitVectorSize; i++){

		unsigned weighPos = (thread * BITS_PER_UNSIGNED) + bit + block_offset;
		thread++;
		interfaceOrderedByBlockSize.setElement(weighPos, bitVector->getElement(i));

		if (thread == block_size){
			thread = 0;
			bit++;
			if (bit == BITS_PER_UNSIGNED){
				bit = 0;
				block_offset += (block_size * BITS_PER_UNSIGNED);
			}
		}
	}
	cuda_copyToDevice(data, interfaceOrderedByBlockSize.getDataPointer(), byteSize);
}

template <VectorType vectorTypeTempl, class c_typeTempl>
CudaVector<vectorTypeTempl, c_typeTempl>::~CudaVector()
{
	if (data) {
		cuda_free(data);
		data = NULL;
	}
}

template <VectorType vectorTypeTempl, class c_typeTempl>
Vector* CudaVector<vectorTypeTempl, c_typeTempl>::clone()
{
	Vector* clone = new CudaVector<vectorTypeTempl, c_typeTempl>(tSize);
	copyTo(clone);
	return clone;
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CudaVector<vectorTypeTempl, c_typeTempl>::copyFromImpl(Interface *interface)
{
	cuda_copyToDevice(data, interface->getDataPointer(), interface->getByteSize());
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CudaVector<vectorTypeTempl, c_typeTempl>::copyToImpl(Interface *interface)
{
	cuda_copyToHost(interface->getDataPointer(), data, this->getByteSize());
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CudaVector<vectorTypeTempl, c_typeTempl>::activation(Vector* resultsVect, FunctionType functionType)
{
	float* results = (float*)resultsVect->getDataPointer();
	cuda_activation(data, tSize, vectorTypeTempl, results, functionType, CUDA_THREADS_PER_BLOCK);
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CudaVector<vectorTypeTempl, c_typeTempl>::mutateImpl(unsigned pos, float mutation)
{
	cuda_mutate(data, pos, mutation, vectorTypeTempl);
}

template <VectorType vectorTypeTempl, class c_typeTempl>
void CudaVector<vectorTypeTempl, c_typeTempl>::crossoverImpl(Vector* other, Interface* bitVector)
{
    CudaVector cudaBitVector = CudaVector(bitVector, Cuda_Threads_Per_Block);

    cuda_crossover(this->getDataPointer(), other->getDataPointer(), (unsigned*)cudaBitVector.getDataPointer(),
						tSize, vectorTypeTempl, Cuda_Threads_Per_Block);
}

template <VectorType vectorTypeTempl, class c_typeTempl>
unsigned CudaVector<vectorTypeTempl, c_typeTempl>::getByteSize()
{
	switch (vectorTypeTempl){
	case BYTE:
		return tSize;
		break;
	case FLOAT:
		return tSize * sizeof(float);
	case BIT:
	case SIGN:
		return (((tSize-1)/BITS_PER_UNSIGNED)+1) * sizeof(unsigned);
	}
}


#endif /* CUDAVECTOR_H_ */
