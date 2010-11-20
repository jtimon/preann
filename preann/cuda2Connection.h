/*
 * cuda2Connection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef CUDA2CONNECTION_H_
#define CUDA2CONNECTION_H_

#include "cudaConnection.h"

class Cuda2Connection: public CudaConnection {
public:
	Cuda2Connection(Vector* input, unsigned outputSize, VectorType vectorType);
	virtual ~Cuda2Connection() {};
	virtual ImplementationType getImplementationType() {
		return CUDA2;
	};

	virtual void addToResults(Vector* results);
};

#endif /* CUDA2CONNECTION_H_ */
