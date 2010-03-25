/*
 * cudaLayer2.h
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#ifndef CUDALAYER2_H_
#define CUDALAYER2_H_

#include "layer.h"
#include "cudaVector.h"

class CudaLayer2: public Layer {
public:
	CudaLayer2(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~CudaLayer2();
};

#endif /* CUDALAYER2_H_ */
