#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "generalDefinitions.h"

#define THREADS_PER_BLOCK 32

typedef struct {
	unsigned numberInputLayers;
	unsigned* inputLayerSize;
	unsigned totalWeighsPerOutput;
	void** inputNeurons;

	unsigned outputSize;
	void* outputNeurons;
	float* thresholds;

	void* weighs;
	FunctionType functionType;
} struct_Layer;

#endif /*CUDA_DEFINITIONS_H_*/
