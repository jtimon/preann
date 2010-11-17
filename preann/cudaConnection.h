/*
 * cudaConnection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDACONNECTION_H_
#define CUDACONNECTION_H_

#include "connection.h"
#include "cudaVector.h"

class CudaConnection: public virtual Connection, public CudaVector {
public:
	CudaConnection(Vector* input, unsigned outputSize, VectorType vectorType);
	virtual ~CudaConnection() {};

	virtual void addToResults(Vector* results);
};

#endif /* CUDACONNECTION_H_ */
