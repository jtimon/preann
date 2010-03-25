#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "generalDefinitions.h"

#define THREADS_PER_BLOCK 64

extern "C" void* cuda_malloc(unsigned byteSize);
extern "C" void cuda_free(void* d_ptr);
extern "C" void cuda_copyToDevice(void* d_dest, void* h_src, unsigned count);
extern "C" void cuda_copyToHost(void* h_dest, void* d_src, unsigned count);
extern "C" float* cuda_getNegativeThresholds(float* thresholds, unsigned size, unsigned block_size);
extern "C" void cuda_activation(void* data, unsigned size, VectorType vectorType, float* results, FunctionType functionType, unsigned block_size);

typedef struct {
	unsigned h_numberInputLayers;
	unsigned* inputLayerSize;
	unsigned* h_inputLayerSize;
	unsigned h_totalWeighsPerOutput;
	void** inputNeurons;

	unsigned h_outputSize;
	void* outputNeurons;
	float* thresholds;

	void* weighs;
	FunctionType h_functionType;
} struct_Layer;

#endif /*CUDA_DEFINITIONS_H_*/
