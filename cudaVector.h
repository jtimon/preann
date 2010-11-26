
#ifndef CUDAVECTOR_H_
#define CUDAVECTOR_H_

#include "vectorImpl.h"
#include "cuda_code.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class CudaVector: virtual public Vector, virtual public VectorImpl<vectorTypeTempl, c_typeTempl> {
private:
	CudaVector() {};
protected:
	unsigned getByteSize()
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

	virtual void copyFromImpl(Interface *interface)
	{
		cuda_copyToDevice(data, interface->getDataPointer(), interface->getByteSize());
	}

	virtual void copyToImpl(Interface *interface)
	{
		cuda_copyToHost(interface->getDataPointer(), data, this->getByteSize());
	}
public:

	virtual ImplementationType getImplementationType() {
		return CUDA;
	};

	CudaVector(unsigned size)
	{
		this->tSize = size;

		unsigned byte_sz = getByteSize();
		data = cuda_malloc(byte_sz);

		cuda_setZero(data, byte_sz, vectorTypeTempl, CUDA_THREADS_PER_BLOCK);
	}
	//special constructor for bit coalescing vectors
	CudaVector(Interface* bitVector, unsigned block_size)
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
	virtual ~CudaVector()
	{
		if (data) {
			cuda_free(data);
			data = NULL;
		}
	}

	virtual Vector* clone()
	{
		Vector* clone = new CudaVector(tSize);
		copyTo(clone);
		return clone;
	}

	virtual void activation(Vector* resultsVect, FunctionType functionType)
	{
		float* results = (float*)resultsVect->getDataPointer();
		cuda_activation(data, tSize, vectorTypeTempl, results, functionType, CUDA_THREADS_PER_BLOCK);
	}

};

#endif /* CUDAVECTOR_H_ */
