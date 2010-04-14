#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "generalDefinitions.h"

#define THREADS_PER_BLOCK 64

void checkCUDAError(const char *msg);


extern "C" void cuda_inputCalculation(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size);
extern "C" void cuda_inputCalculation2(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size);
extern "C" void cuda_inputCalculation3(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size);



#endif /*CUDA_DEFINITIONS_H_*/
