
#ifndef CUDAINVERTEDCONNECTION_H_
#define CUDAINVERTEDCONNECTION_H_

#include "connection.h"
#include "cudaVector.h"

class CudaInvertedConnection: public virtual Connection, public CudaVector {
protected:
	//redefined from CudaVector
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
public:
	CudaInvertedConnection(Vector* input, unsigned outputSize, VectorType vectorType);
	virtual ~CudaInvertedConnection() {};

	virtual void mutate(unsigned pos, float mutation);
	virtual void crossover(Connection* other, Interface* bitVector);
	virtual void addToResults(Vector* results);

};

#endif /* CUDAINVERTEDCONNECTION_H_ */
