#ifndef CUDAINVERTEDCONNECTION_H_
#define CUDAINVERTEDCONNECTION_H_

#include "connection.h"
#include "cudaVector.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class CudaInvertedConnection: public virtual Connection, public CudaVector<vectorTypeTempl, c_typeTempl> {
protected:
	//redefined from CudaVector
	virtual void copyFromImpl(Interface* interface)
	{
		interface->transposeMatrix(tInput->getSize());
		CudaVector<vectorTypeTempl, c_typeTempl>::copyFromImpl(interface);
	}

	virtual void copyToImpl(Interface* interface)
	{
		CudaVector<vectorTypeTempl, c_typeTempl>::copyToImpl(interface);
		interface->transposeMatrix(tSize / tInput->getSize());
	}

	virtual void mutateImpl(unsigned pos, float mutation)
	{
		//TODO z simplificar cuentas
		unsigned outputPos = pos / tInput->getSize();
		unsigned inputPos = (pos % tInput->getSize());
		unsigned outputSize = tSize / tInput->getSize();
		pos = outputPos + (inputPos * outputSize);

		cuda_mutate(data, pos, mutation, vectorTypeTempl);
	}

	virtual void crossoverImpl(Vector* other, Interface* bitVector)
	{
		Interface invertedBitVector = Interface(bitVector);
		invertedBitVector.transposeMatrix(tInput->getSize());

		CudaVector<vectorTypeTempl, c_typeTempl>::crossoverImpl(other, &invertedBitVector);
	}
public:
	CudaInvertedConnection(Vector* input, unsigned outputSize)
		: CudaVector<vectorTypeTempl, c_typeTempl>(input->getSize() * outputSize)
	{
		tInput = input;
	}

	virtual ~CudaInvertedConnection() {};

	virtual ImplementationType getImplementationType() {
		return CUDA_INV;
	};

	virtual void calculateAndAddTo(Vector* results)
	{
		void* inputWeighs = this->getDataPointer();
		float* resultsPtr = (float*)results->getDataPointer();

		cuda_inputCalculationInvertedMatrix(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
	}

};

#endif /* CUDAINVERTEDCONNECTION_H_ */
