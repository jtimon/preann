#ifndef CUDA_DEFINITIONS_H_
#define CUDA_DEFINITIONS_H_

#include "generalDefinitions.h"

#define THREADS_PER_BLOCK 64

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
