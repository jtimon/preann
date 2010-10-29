/*
 * cudaLayer2.h
 *
 *  Created on: Apr 15, 2010
 *      Author: timon
 */

#ifndef CUDALAYER2_H_
#define CUDALAYER2_H_

#include "cudaLayer.h"

class CudaLayer2: public CudaLayer {
protected:
public:
	CudaLayer2() {};
	virtual ~CudaLayer2() {};
	virtual ImplementationType getImplementationType() {
		return CUDA2;
	};

	virtual void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector);
};

#endif /* CUDALAYER2_H_ */
