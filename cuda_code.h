#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "generalDefinitions.h"

#define THREADS_PER_BLOCK 64

void checkCUDAError(const char *msg);

extern "C" void cuda_setZero(void* data, unsigned byteSize, VectorType vectorType, unsigned block_size);
extern "C" float* cuda_getNegativeThresholds(float* thresholds, unsigned size, unsigned block_size);
extern "C" void cuda_inputCalculation(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size);
extern "C" void cuda_inputCalculation2(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size);
extern "C" void cuda_inputCalculation3(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size);
extern "C" void cuda_activation(void* data, unsigned size, VectorType vectorType, float* results, FunctionType functionType, unsigned block_size);


#endif /*CUDA_DEFINITIONS_H_*/
