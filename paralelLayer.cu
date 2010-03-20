#include "cudaDefinitions.h"

__device__ float Func(float number, FunctionType functionType) {

	//printf("number: %f ", number);
	switch (functionType) {

		//TODO añadir diferentes funciones

		case BINARY_STEP:
			if (number > 0){
				return 1;
			} else {
				return 0;
			}
		case BIPOLAR_STEP:
			if (number > 0){
				return 1;
			} else {
				return -1;
			}
		//case ANOTHER_FUNCTION:
		//	return anotherFunction(number);

		case IDENTITY:
		default:
			return number;
	}
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        printf("Cuda error: %s : %s.\n", msg,
                                  cudaGetErrorString( err) );
	//printf("Cuda error: %s ( %d ): %s.\n", msg, __LINE__, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

extern "C" struct_Layer* LayerHostToDevice(struct_Layer* h_layer, VectorType inputType, VectorType outputType){

	size_t size;
	struct_Layer* d_layer = (struct_Layer*) mi_malloc(sizeof(struct_Layer));

	d_layer->h_numberInputLayers = h_layer->h_numberInputLayers;
	d_layer->h_totalWeighsPerOutput = h_layer->h_totalWeighsPerOutput;
	d_layer->h_outputSize = h_layer->h_outputSize;
	d_layer->h_functionType = h_layer->h_functionType;

	size = h_layer->h_numberInputLayers * sizeof(unsigned);
	//TODO
	d_layer->h_inputLayerSize = (unsigned*)mi_malloc(size);
	memcpy(d_layer->h_inputLayerSize, h_layer->inputLayerSize, size);

	cudaMalloc((void**)&(d_layer->inputLayerSize), size);
	cudaMemcpy(d_layer->inputLayerSize, h_layer->inputLayerSize, size, cudaMemcpyHostToDevice);
//TODO
	size = h_layer->h_numberInputLayers * sizeof(void*);
	cudaMalloc((void**)&(d_layer->inputNeurons), size);

	if (outputType == FLOAT){
		size = sizeof(float) * h_layer->h_outputSize * h_layer->h_totalWeighsPerOutput;
	} else {
		size = sizeof(unsigned char) * h_layer->h_outputSize * h_layer->h_totalWeighsPerOutput;
	}
	cudaMalloc((void**)&(d_layer->weighs), size);
	cudaMemcpy(d_layer->weighs, h_layer->weighs, size, cudaMemcpyHostToDevice);

	if (outputType == FLOAT){
		size = sizeof(float) * h_layer->h_outputSize;
	} else {
		size = sizeof(unsigned) * (((h_layer->h_outputSize - 1)/ BITS_PER_UNSIGNED) + 1);
	}
	cudaMalloc((void**)&(d_layer->outputNeurons), size);
	cudaMemcpy(d_layer->outputNeurons, h_layer->outputNeurons, size, cudaMemcpyHostToDevice);

	size = h_layer->h_outputSize * sizeof(float);
	cudaMalloc((void**)&(d_layer->thresholds), size);
	cudaMemcpy(d_layer->thresholds, h_layer->thresholds, size, cudaMemcpyHostToDevice);
	
	checkCUDAError("Layer Host To Device");
	return d_layer;
}

extern "C" void FreeDevice(struct_Layer* d_layer){

	cudaFree(d_layer->inputLayerSize);
	cudaFree(d_layer->inputNeurons);
	cudaFree(d_layer->outputNeurons);
	cudaFree(d_layer->weighs);
	cudaFree(d_layer->thresholds);

	mi_free(d_layer);

	checkCUDAError("Free Device");
}

extern "C" void SetInputsInDevice(struct_Layer* d_layer, void** inputs){

	size_t size = sizeof(void*) * d_layer->h_numberInputLayers;
	cudaMemcpy(d_layer->inputNeurons, inputs, size, cudaMemcpyHostToDevice);

	checkCUDAError("Set Inputs In Device");
}

extern "C" void** InputsToDevice(void** host_inputs, unsigned* host_inputSizes, VectorType* host_types, unsigned numberInputs)
{
	size_t size = numberInputs * sizeof(void*);
	void** dev_inputs;
	dev_inputs = (void**) mi_malloc(sizeof(void*) * numberInputs);
	
	for (unsigned i=0; i < numberInputs; i++){
		
		if (host_types[i] == FLOAT){
		
			size = host_inputSizes[i] * sizeof(float);
		} else {
			size = (((host_inputSizes[i] - 1)/ BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
		}
		cudaMalloc((void**)&(dev_inputs[i]), size);
		cudaMemcpy(dev_inputs[i], host_inputs[i], size, cudaMemcpyHostToDevice);
	}
	checkCUDAError("Inputs To Device");
	return dev_inputs;
}


extern "C" void FreeInputs(void** dev_inputs, unsigned numberInputs)
{
	for (unsigned i=0; i < numberInputs; i++){
		cudaFree(dev_inputs[i]);
	}
	mi_free(dev_inputs);
	checkCUDAError("Free Inputs");
}

extern "C" void RefreshDeviceInputs(void** dev_inputs, void** host_inputs, unsigned* host_inputSizes, VectorType* host_types, unsigned numberInputs)
{
	size_t size;
	for (unsigned i=0; i < numberInputs; i++){
		
		if (host_types[i] == FLOAT){
		
			size = host_inputSizes[i] * sizeof(float);
		} else {
			size = (((host_inputSizes[i] - 1)/ BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
		}
		cudaMemcpy(dev_inputs[i], host_inputs[i], size, cudaMemcpyHostToDevice);
	}
	checkCUDAError("Refresh Device Inputs");
}

extern "C" void OutputToHost(void* output, struct_Layer* d_layer, VectorType outputType){

	size_t size;

	if (outputType == FLOAT){
		size = sizeof(float) * d_layer->h_outputSize;
	} else {
		size = sizeof(unsigned) * (((d_layer->h_outputSize - 1)/ BITS_PER_UNSIGNED) + 1);
	}	cudaMemcpy(output, d_layer->outputNeurons, size, cudaMemcpyDeviceToHost);
	checkCUDAError("Output To Host");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//NEW VERSION

template <unsigned int blockSize, VectorType inputType>
__global__ void SumConnectionsKernel(struct_Layer layer, unsigned input_id, unsigned input_size, unsigned inputOffset, float* results)
{
	extern __shared__ float sdata[];

	unsigned tid = threadIdx.x;
	unsigned outputNeuron = blockIdx.x;
	unsigned weighsOffset = (outputNeuron * layer.h_totalWeighsPerOutput);

	float result = 0;
	unsigned i = tid;

	if (inputType == FLOAT) {
		weighsOffset += inputOffset;
		while (i < input_size){
			if (inputType == FLOAT){
				result += ((float**)(layer.inputNeurons))[input_id][i] * ((float*)layer.weighs)[weighsOffset + i];
				i += blockSize;
			}
		}
	} else {
		weighsOffset += inputOffset * BITS_PER_UNSIGNED;
		unsigned elementsToRead = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
		while (i < elementsToRead){

			unsigned mask = 0x80000000;
			unsigned currentInput = ((unsigned**)(layer.inputNeurons))[input_id][i];
			for (unsigned j=0; j < BITS_PER_UNSIGNED; j++) {
				//printf(" %d", ((unsigned char*)layer.weighs)[weighsOffset + j] - 128);
				if (currentInput & mask) {
					//printf("X", 1);
					result += ((unsigned char*)layer.weighs)[weighsOffset + j] - 128;
				} else if (inputType == SIGN) {
					//printf(" b ", 1);
					result += 128 - ((unsigned char*)layer.weighs)[weighsOffset + j];
				}
				//printf(" ", 1);
				//printf(" mask %d ", mask);
				mask >>= 1;
			}
			i += blockSize;
			weighsOffset += blockSize * BITS_PER_UNSIGNED;
		}
	}
	sdata[tid] = result;
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) {  sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

#if __DEVICE_EMULATION__
	if (blockSize >= 64) { if (tid < 32) { sdata[tid] += sdata[tid + 32]; } __syncthreads(); }
	if (blockSize >= 32) { if (tid < 16) { sdata[tid] += sdata[tid + 16]; } __syncthreads(); }
	if (blockSize >= 16) { if (tid < 8) {  sdata[tid] += sdata[tid + 8]; }  __syncthreads(); }
	if (blockSize >= 8) {  if (tid < 4) {  sdata[tid] += sdata[tid + 4]; }  __syncthreads(); }
	if (blockSize >= 4) {  if (tid < 2) {  sdata[tid] += sdata[tid + 2]; }  __syncthreads(); }
	if (blockSize >= 2) {  if (tid < 1) {  sdata[tid] += sdata[tid + 1]; }  __syncthreads(); }
#else
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
	}
#endif
	if (tid == 0) {
		results[outputNeuron] += sdata[0];
	}
}

void SumLayerConnections(struct_Layer* layer, float* d_results, unsigned block_size, VectorType inputType){

	dim3 dimBlock(block_size, 1, 1);
	dim3 dimGrid(layer->h_outputSize, 1, 1);
	int smemSize = block_size * sizeof(float);


	unsigned inputOffset = 0;
	for (unsigned i=0; i < layer->h_numberInputLayers; i++){
		if (inputType == FLOAT){
			switch (block_size)
			{
				case 512:
					SumConnectionsKernel<512, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 256:
					SumConnectionsKernel<256, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 128:
					SumConnectionsKernel<128, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 64:
					SumConnectionsKernel< 64, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 32:
					SumConnectionsKernel< 32, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 16:
					SumConnectionsKernel< 16, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  8:
					SumConnectionsKernel<  8, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  4:
					SumConnectionsKernel<  4, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  2:
					SumConnectionsKernel<  2, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  1:
					SumConnectionsKernel<  1, FLOAT><<<dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
			}
		} else if (inputType == BIT) {
			switch (block_size)
			{
				case 512:
					SumConnectionsKernel<512, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 256:
					SumConnectionsKernel<256, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 128:
					SumConnectionsKernel<128, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 64:
					SumConnectionsKernel< 64, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 32:
					SumConnectionsKernel< 32, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 16:
					SumConnectionsKernel< 16, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  8:
					SumConnectionsKernel<  8, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  4:
					SumConnectionsKernel<  4, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  2:
					SumConnectionsKernel<  2, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  1:
					SumConnectionsKernel<  1, BIT><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
			}
		} else {
			switch (block_size)
			{
				case 512:
					SumConnectionsKernel<512, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 256:
					SumConnectionsKernel<256, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 128:
					SumConnectionsKernel<128, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 64:
					SumConnectionsKernel< 64, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 32:
					SumConnectionsKernel< 32, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case 16:
					SumConnectionsKernel< 16, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  8:
					SumConnectionsKernel<  8, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  4:
					SumConnectionsKernel<  4, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  2:
					SumConnectionsKernel<  2, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
				case  1:
					SumConnectionsKernel<  1, SIGN><<< dimGrid, dimBlock, smemSize >>>(*layer, i, layer->h_inputLayerSize[i], inputOffset, d_results); break;
			}
		}
		inputOffset += layer->h_inputLayerSize[i];
	}
	checkCUDAError("SumLayerConnections");
}

/*
__global__ void set_float_array(float* array, float value, unsigned array_sz)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < array_sz) array[idx] = value;
}*/

__global__ void negative_thresholds_kernel(float* results, float* thresholds, unsigned results_sz)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < results_sz) results[idx] = - thresholds[idx];
}

__global__ void activation_float_kernel(float* results, float* output, unsigned output_sz, FunctionType functionType)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < output_sz) output[idx] = Func(results[idx], functionType);
}

__global__ void activation_bit_kernel(float* results, unsigned* output, unsigned output_sz)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned offset = idx * BITS_PER_UNSIGNED;
	unsigned toRead;
	if (output_sz > offset) {
		toRead = output_sz - offset;
		if (toRead > BITS_PER_UNSIGNED) {
			toRead = BITS_PER_UNSIGNED;
		}
	} else {
		toRead = 0;
	}

	if (toRead > 0){

		unsigned threadOutput = 0;
		unsigned mask = 0x80000000;
		for (unsigned i=0; i < toRead; i++){
			if (results[offset + i] > 0){
				threadOutput |= mask;
			} else {
				threadOutput &= ~mask;
			}
			mask >>= 1;
		}
		output[idx] = threadOutput;
	}
}

extern "C" void LayerCalculation2(struct_Layer* d_layer, unsigned block_size, VectorType inputType, VectorType outputType){

	unsigned grid_size = ((d_layer->h_outputSize - 1)/block_size) + 1;

	float* results;
	size_t size = sizeof(float) * d_layer->h_outputSize;
	cudaMalloc((void**)&(results), size);

	negative_thresholds_kernel<<< grid_size, block_size >>>(results, d_layer->thresholds, d_layer->h_outputSize);

	SumLayerConnections(d_layer, results, block_size, inputType);

	for (unsigned i=0; i < d_layer->h_outputSize; i++){
		printf(" %f ", results[i]);
	}
	printf("\n ", 1);

	if (outputType == FLOAT) {
		activation_float_kernel<<< grid_size, block_size >>>(results, (float*)d_layer->outputNeurons, d_layer->h_outputSize, d_layer->h_functionType);
	} else {

		grid_size = ((d_layer->h_outputSize - 1) / (block_size * BITS_PER_UNSIGNED)) + 1;
		//printf("bloques %d / block_size %d / outputSize %d \n", grid_size, block_size, d_layer->h_outputSize);
		activation_bit_kernel<<< grid_size, block_size >>>(results, (unsigned*)d_layer->outputNeurons, d_layer->h_outputSize);
	}
	checkCUDAError("LayerCalculation2");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//OLD VERSION

template <unsigned int blockSize, VectorType inputType, VectorType outputType>
__global__ void LayerCalculationKernel(struct_Layer layer)
{
	extern __shared__ float sdata[];

	unsigned tid = threadIdx.x;
	unsigned outputNeuron = blockIdx.x;
	unsigned weighsOffset = (outputNeuron * layer.h_totalWeighsPerOutput);

	float result = 0;
	for (unsigned input=0; input < layer.h_numberInputLayers; input++){

		unsigned i = tid;
		unsigned elementsToRead;
		if (inputType == FLOAT){
			elementsToRead = layer.inputLayerSize[input];
		} else {
			elementsToRead = ((layer.inputLayerSize[input] - 1) / BITS_PER_UNSIGNED) + 1;
		}

		unsigned mask = 0x80000000;
		while (i < elementsToRead){
			if (inputType == FLOAT){
				result += ((float**)(layer.inputNeurons))[input][i] * ((float*)layer.weighs)[weighsOffset + i];
			}
			if (inputType == BIT){

				for (unsigned j=0; j < BITS_PER_UNSIGNED; j++) {
					if (((unsigned**)(layer.inputNeurons))[input][i] & mask) {
						result += ((unsigned char*)layer.weighs)[weighsOffset + (i * BITS_PER_UNSIGNED) + j] - 128;
					}
					mask >>= 1;
				}
				mask = 0x80000000;
			}
			if (inputType == SIGN){

				for (unsigned j=0; j < BITS_PER_UNSIGNED; j++) {
					if (((unsigned**)(layer.inputNeurons))[input][i] & mask) {
						result += ((unsigned char*)layer.weighs)[weighsOffset + (i * BITS_PER_UNSIGNED) + j] - 128;
					} else {
						result += 128 - ((unsigned char*)layer.weighs)[weighsOffset + (i * BITS_PER_UNSIGNED) + j];
					}
					mask >>= 1;
				}
				mask = 0x80000000;
			}
			i += blockSize;
		}
		if (inputType == FLOAT){
			weighsOffset += elementsToRead;
		} else {
			weighsOffset += elementsToRead * BITS_PER_UNSIGNED;
		}
	}

	sdata[tid] = result; //TODO quizá no almacene result en un registro y sea más práctico usar sdata[tid], en su lugar, directamente
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

#if __DEVICE_EMULATION__
	if (blockSize >= 64) { if (tid < 32) { sdata[tid] += sdata[tid + 32]; } __syncthreads(); }
	if (blockSize >= 32) { if (tid < 16) { sdata[tid] += sdata[tid + 16]; } __syncthreads(); }
	if (blockSize >= 16) { if (tid < 8) { sdata[tid] += sdata[tid + 8]; } __syncthreads(); }
	if (blockSize >= 8) { if (tid < 4) { sdata[tid] += sdata[tid + 4]; } __syncthreads(); }
	if (blockSize >= 4) { if (tid < 2) { sdata[tid] += sdata[tid + 2]; } __syncthreads(); }
	if (blockSize >= 2) { if (tid < 1) { sdata[tid] += sdata[tid + 1]; } __syncthreads(); }
#else
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
#endif
	if (outputType == FLOAT) {
		if (tid == 0) {
			((float*)(layer.outputNeurons))[outputNeuron] = Func(sdata[0] - layer.thresholds[outputNeuron], layer.h_functionType);
		}
	}
	if (outputType == BIT) {
		if (tid == 0) {
			//printf(" %d ", (int)sdata[0]);
			unsigned mask = (unsigned)(0x80000000>>(outputNeuron % BITS_PER_UNSIGNED));
			if (sdata[0] - layer.thresholds[outputNeuron] > 0){
				atomicOr(&(((unsigned*)(layer.outputNeurons))[outputNeuron / BITS_PER_UNSIGNED]), mask);
			} else {
				atomicAnd(&(((unsigned*)(layer.outputNeurons))[outputNeuron / BITS_PER_UNSIGNED]), ~mask);
			}
		}
	}
}

extern "C" void LayerCalculation(struct_Layer* d_layer, unsigned threads, VectorType inputType, VectorType outputType){

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(d_layer->h_outputSize, 1, 1);
	int smemSize = threads * sizeof(float);
	switch (inputType) {
		case FLOAT:
			if (outputType == FLOAT){
				switch (threads)
		        {
			        case 512:
			            LayerCalculationKernel<512, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 256:
			            LayerCalculationKernel<256, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 128:
			            LayerCalculationKernel<128, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 64:
			            LayerCalculationKernel< 64, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 32:
			            LayerCalculationKernel< 32, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 16:
			            LayerCalculationKernel< 16, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  8:
			            LayerCalculationKernel<  8, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  4:
			            LayerCalculationKernel<  4, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  2:
			            LayerCalculationKernel<  2, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  1:
			            LayerCalculationKernel<  1, FLOAT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
		        }
			} else {
				switch (threads)
		        {
			        case 512:
			            LayerCalculationKernel<512, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 256:
			            LayerCalculationKernel<256, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 128:
			            LayerCalculationKernel<128, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 64:
			            LayerCalculationKernel< 64, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 32:
			            LayerCalculationKernel< 32, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 16:
			            LayerCalculationKernel< 16, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  8:
			            LayerCalculationKernel<  8, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  4:
			            LayerCalculationKernel<  4, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  2:
			            LayerCalculationKernel<  2, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  1:
			            LayerCalculationKernel<  1, FLOAT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
		        }
			}
			break;
		case BIT:
			if (outputType == FLOAT){
				switch (threads)
		        {
			        case 512:
			            LayerCalculationKernel<512, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 256:
			            LayerCalculationKernel<256, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 128:
			            LayerCalculationKernel<128, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 64:
			            LayerCalculationKernel< 64, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 32:
			            LayerCalculationKernel< 32, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 16:
			            LayerCalculationKernel< 16, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  8:
			            LayerCalculationKernel<  8, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  4:
			            LayerCalculationKernel<  4, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  2:
			            LayerCalculationKernel<  2, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  1:
			            LayerCalculationKernel<  1, BIT, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
		        }
			} else {
				switch (threads)
		        {
			        case 512:
			            LayerCalculationKernel<512, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 256:
			            LayerCalculationKernel<256, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 128:
			            LayerCalculationKernel<128, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 64:
			            LayerCalculationKernel< 64, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 32:
			            LayerCalculationKernel< 32, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 16:
			            LayerCalculationKernel< 16, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  8:
			            LayerCalculationKernel<  8, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  4:
			            LayerCalculationKernel<  4, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  2:
			            LayerCalculationKernel<  2, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  1:
			            LayerCalculationKernel<  1, BIT, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
		        }
			}
			break;
		case SIGN:
			if (outputType == FLOAT){
				switch (threads)
		        {
			        case 512:
			            LayerCalculationKernel<512, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 256:
			            LayerCalculationKernel<256, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 128:
			            LayerCalculationKernel<128, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 64:
			            LayerCalculationKernel< 64, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 32:
			            LayerCalculationKernel< 32, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 16:
			            LayerCalculationKernel< 16, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  8:
			            LayerCalculationKernel<  8, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  4:
			            LayerCalculationKernel<  4, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  2:
			            LayerCalculationKernel<  2, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  1:
			            LayerCalculationKernel<  1, SIGN, FLOAT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
		        }
			} else {
				switch (threads)
		        {
			        case 512:
			            LayerCalculationKernel<512, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 256:
			            LayerCalculationKernel<256, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 128:
			            LayerCalculationKernel<128, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 64:
			            LayerCalculationKernel< 64, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 32:
			            LayerCalculationKernel< 32, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case 16:
			            LayerCalculationKernel< 16, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  8:
			            LayerCalculationKernel<  8, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  4:
			            LayerCalculationKernel<  4, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  2:
			            LayerCalculationKernel<  2, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
			        case  1:
			            LayerCalculationKernel<  1, SIGN, BIT><<< dimGrid, dimBlock, smemSize >>>(*d_layer); break;
		        }
			}
			break;
	}
	checkCUDAError("Layer Calculation");
}

