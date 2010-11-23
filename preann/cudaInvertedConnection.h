#ifndef CUDAINVERTEDCONNECTION_H_
#define CUDAINVERTEDCONNECTION_H_

#include "connection.h"
#include "cudaVector.h"

template <VectorType vectorTypeTempl>
class CudaInvertedConnection: public virtual Connection, public CudaVector<vectorTypeTempl> {
protected:
	//redefined from CudaVector
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);
public:
	CudaInvertedConnection(Vector* input, unsigned outputSize);
	virtual ~CudaInvertedConnection() {};
	virtual ImplementationType getImplementationType() {
		return CUDA_INV;
	};

	virtual void calculateAndAddTo(Vector* results);

};

template <VectorType vectorTypeTempl>
CudaInvertedConnection<vectorTypeTempl>::CudaInvertedConnection(Vector* input, unsigned outputSize)
	: CudaVector<vectorTypeTempl>(input->getSize() * outputSize)
{
	tInput = input;
}

template <VectorType vectorTypeTempl>
void CudaInvertedConnection<vectorTypeTempl>::mutateImpl(unsigned pos, float mutation)
{
	//TODO z simplificar cuentas
	unsigned outputPos = pos / tInput->getSize();
	unsigned inputPos = (pos % tInput->getSize());
	unsigned outputSize = tSize / tInput->getSize();
	pos = outputPos + (inputPos * outputSize);

	cuda_mutate(data, pos, mutation, vectorTypeTempl);
}

template <VectorType vectorTypeTempl>
void CudaInvertedConnection<vectorTypeTempl>::crossoverImpl(Vector* other, Interface* bitVector)
{
	Interface invertedBitVector = Interface(bitVector);
	invertedBitVector.transposeMatrix(tInput->getSize());

	CudaVector<vectorTypeTempl>::crossoverImpl(other, &invertedBitVector);
}

template <VectorType vectorTypeTempl>
void CudaInvertedConnection<vectorTypeTempl>::calculateAndAddTo(Vector* results)
{
	void* inputWeighs = this->getDataPointer();
	float* resultsPtr = (float*)results->getDataPointer();

	cuda_inputCalculationInvertedMatrix(tInput->getDataPointer(), tInput->getSize(), tInput->getVectorType(), results->getSize(), inputWeighs, resultsPtr, Cuda_Threads_Per_Block);
}

template <VectorType vectorTypeTempl>
void CudaInvertedConnection<vectorTypeTempl>::copyFromImpl(Interface* interface)
{
	interface->transposeMatrix(tInput->getSize());
	CudaVector<vectorTypeTempl>::copyFromImpl(interface);
}

template <VectorType vectorTypeTempl>
void CudaInvertedConnection<vectorTypeTempl>::copyToImpl(Interface* interface)
{
	CudaVector<vectorTypeTempl>::copyToImpl(interface);
	interface->transposeMatrix(tSize / tInput->getSize());
}


#endif /* CUDAINVERTEDCONNECTION_H_ */
