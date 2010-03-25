/*
 * cudaLayer2.cpp
 *
 *  Created on: Mar 25, 2010
 *      Author: timon
 */

#include "cudaLayer2.h"

CudaLayer2::CudaLayer2(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
	// TODO Auto-generated constructor stub
}

CudaLayer2::~CudaLayer2()
{
	// TODO Auto-generated destructor stub
}

Layer* CudaLayer2::newCopy()
{
	std::string error = "save is not implemented for newCopy.";
	throw error;
}

void CudaLayer2::save(FILE *stream)
{
	std::string error = "save is not implemented for CudaLayer2.";
	throw error;
}

void CudaLayer2::load(FILE *stream)
{
	std::string error = "load is not implemented for CudaLayer2.";
	throw error;
}

void CudaLayer2::randomWeighs(float range)
{
	std::string error = "randomWeighs is not implemented for CudaLayer2.";
	throw error;
}

Vector* CudaLayer2::newVector(unsigned  size, VectorType vectorType)
{
	return new CudaVector(size, vectorType);
}

void CudaLayer2::calculateOutput()
{
	float* results = cuda_getNegativeThresholds(thresholds, output->getSize(), THREADS_PER_BLOCK);

	for(unsigned i=0; i < numberInputs; i++){
		//TODO
	}
	//cuda_calculateOutput(results, output->getDataPointer(), output->getSize(), )

	output->activation(results, functionType);
}


