/*
 * cudaLayer2.h
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#ifndef CUDALAYER_H_
#define CUDALAYER_H_

#include "layer.h"
//TODO quitar el constructor especial de cudaVector y este include (cambiarlo por cuda_code)
#include "cudaVector.h"

class CudaLayer: public Layer {
public:
	CudaLayer() {};
	virtual ~CudaLayer() {};
	virtual ImplementationType getImplementationType() {
		return CUDA;
	};

	virtual void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector);

};

#endif /* CUDALAYER_H_ */
