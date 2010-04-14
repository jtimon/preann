
#include "cuda_code.h"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        printf("Cuda error: %s : %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

// LAYER CALCULATION

__global__
void SumFloatsConnectionsKernel(float* inputs, unsigned input_size, unsigned output_size, float* weighs, float* results)
{
	extern __shared__ float sdata[];

	unsigned outputNeuron = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned weighsOffset = outputNeuron * input_size;
	float result = 0;

	unsigned pos = threadIdx.x;
	while (pos < input_size){

		sdata[pos] = inputs[pos];
		pos += blockDim.x;
	}
	__syncthreads();

	if (outputNeuron < output_size){

		//////////////////////////
		for (unsigned i=0; i < input_size; i++){
			result += sdata[i] * weighs[weighsOffset + i];
			//printf(" peso %f ", weighs[weighsOffset + i]);
		}
		/////TODO OTRA OPCION
	/*	if (blockDim.x <= input_size){
			unsigned pos = tid;
			while (pos < input_size){
				result += sdata[pos] * weighs[weighsOffset + pos];
				++pos;
			}
			pos = 0;
			while (pos < tid){
				result += sdata[pos] * weighs[weighsOffset + pos];
				++pos;
			}
		} else {
			unsigned pos = tid;
			while (pos < input_size){
				result += sdata[pos] * weighs[weighsOffset + pos];
				++pos;
			}
			unsigned newMax = min(tid, input_size);
			pos = 0;
			while (pos < newMax){
				result += sdata[pos] * weighs[weighsOffset + pos];
				++pos;
			}
		}*/
		/////////////
		results[outputNeuron] += result;
	}
}

template <VectorType inputType>
__global__
void SumBitsConnectionsKernel(unsigned* inputs, unsigned input_size, unsigned output_size, unsigned char* weighs, float* results)
{
	extern __shared__ unsigned shared_inputs[];

	unsigned tid = threadIdx.x;
	unsigned input_blocks_to_read = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
	unsigned readingLoops = ((input_blocks_to_read - 1) / blockDim.x) + 1;

	unsigned pos = tid;

	for (unsigned i=0; i < readingLoops; i++){
		if (pos < input_blocks_to_read){
			shared_inputs[pos] = inputs[pos];
		}
		pos += blockDim.x;
	}
	__syncthreads();

	unsigned outputNeuron = blockIdx.x*blockDim.x + threadIdx.x;
	if (outputNeuron < output_size){

		float result = 0;
		unsigned weighsOffset = (outputNeuron * input_size);

		for (unsigned i=0; i < input_blocks_to_read; i++){

			unsigned input_block = shared_inputs[i];
			unsigned mask = 0x80000000;
			for (unsigned j=0; j < BITS_PER_UNSIGNED; j++){

				if (input_block & mask){
					result += weighs[weighsOffset] - 128;
				} else {
					if (inputType == SIGN) {
						result += 128 - weighs[weighsOffset];
					}
				}
				++weighsOffset;
				mask >>= 1;
			}
		}
		results[outputNeuron] += result;
	}
}

extern "C" void cuda_inputCalculation(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size)
{
	unsigned grid_size = ((output_size - 1)/block_size) + 1;
	unsigned shared_mem_size;

	if (inputType == FLOAT) {
		//TODO quitar estas comprobaciones y hacer que sirva para cualquier tamaño de entrada
		if (input_size > 4032){
			string error = "The maximum float input size is 4032.";
			throw error;
		}
		shared_mem_size = input_size * sizeof(float);

		SumFloatsConnectionsKernel<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtr, input_size, output_size, (float*)weighs, results);
	} else {

		shared_mem_size =(((input_size - 1)/BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
		//TODO quitar estas comprobaciones y hacer que sirva para cualquier tamaño de entrada
		if (shared_mem_size > 16128){
			//16128 * 8
			string error = "The maximum bit/sign input size is 129024.";
			throw error;
		}
		if (inputType == BIT) {
			SumBitsConnectionsKernel<BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		} else {
			SumBitsConnectionsKernel<SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		}
	}
}

__global__
void SumFloatsConnectionsKernel3(float* inputs, unsigned input_size, unsigned input_id, unsigned output_size, float* weighs, float* results)
{
	extern __shared__ float sdata[];

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx == 0) sdata[0] = inputs[input_id];
	__syncthreads();

	if (idx < output_size) results[idx] += sdata[0] * weighs[(idx * input_size) + input_id];
}

extern "C" void cuda_inputCalculation3(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size)
{
	unsigned grid_size = ((output_size - 1)/block_size) + 1;
	unsigned shared_mem_size;

	if (inputType == FLOAT) {

		shared_mem_size = sizeof(float);
		for (unsigned i=0; i < input_size; i++) {

			SumFloatsConnectionsKernel3<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtr, input_size, i, output_size, (float*)weighs, results);
		}
	} else {

		shared_mem_size =(((input_size - 1)/BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
		//TODO quitar estas comprobaciones y hacer que sirva para cualquier tamaño de entrada
		if (shared_mem_size > 16128){
			//16128 * 8
			string error = "The maximum bit/sign input size is 129024.";
			throw error;
		}
		if (inputType == BIT) {
			SumBitsConnectionsKernel<BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		} else {
			SumBitsConnectionsKernel<SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		}
	}
}

template <unsigned int blockSize, VectorType inputType>
__global__
void SumConnectionsKernel(void* inputPtr, unsigned input_size, unsigned output_size, void* weighs, float* results)
{
	extern __shared__ float sdata[];

	unsigned tid = threadIdx.x;
	unsigned outputNeuron = blockIdx.x;
	unsigned weighsOffset = (outputNeuron * input_size);

	float result = 0;
	unsigned i = tid;

	if (inputType == FLOAT) {
		while (i < input_size){
			if (inputType == FLOAT){
				result += ((float*)inputPtr)[i] * ((float*)weighs)[weighsOffset + i];
				i += blockSize;
			}
		}
	} else {
		weighsOffset += tid * BITS_PER_UNSIGNED;

		unsigned input_blocks_to_read = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
		while (i < input_blocks_to_read){

			unsigned mask = 0x80000000;
			unsigned currentInput = ((unsigned*)inputPtr)[i];
			for (unsigned j=0; j < BITS_PER_UNSIGNED; j++) {

				if (currentInput & mask) {
					result += ((unsigned char*)weighs)[weighsOffset + j] - 128;
				} else {
					if (inputType == SIGN) {
						result += 128 - ((unsigned char*)weighs)[weighsOffset + j];
					}
				}
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

//void SumLayerConnections(struct_Layer* layer, float* d_results, unsigned block_size, VectorType inputType){
extern "C" void cuda_inputCalculation2(void* inputPtr, unsigned input_size, VectorType inputType, unsigned output_size, void* weighs, float* results, unsigned block_size)
{
	unsigned grid_size = output_size;
	unsigned shared_mem_size = block_size * sizeof(float);

	if (inputType == FLOAT){
		switch (block_size)
		{
			case 512:
				SumConnectionsKernel<512, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 256:
				SumConnectionsKernel<256, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 128:
				SumConnectionsKernel<128, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 64:
				SumConnectionsKernel< 64, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 32:
				SumConnectionsKernel< 32, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 16:
				SumConnectionsKernel< 16, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  8:
				SumConnectionsKernel<  8, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  4:
				SumConnectionsKernel<  4, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  2:
				SumConnectionsKernel<  2, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  1:
				SumConnectionsKernel<  1, FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		}
	} else if (inputType == BIT) {
		switch (block_size)
		{
			case 512:
				SumConnectionsKernel<512, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 256:
				SumConnectionsKernel<256, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 128:
				SumConnectionsKernel<128, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 64:
				SumConnectionsKernel< 64, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 32:
				SumConnectionsKernel< 32, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 16:
				SumConnectionsKernel< 16, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  8:
				SumConnectionsKernel<  8, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  4:
				SumConnectionsKernel<  4, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  2:
				SumConnectionsKernel<  2, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  1:
				SumConnectionsKernel<  1, BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		}
	} else {
		switch (block_size)
		{
			case 512:
				SumConnectionsKernel<512, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 256:
				SumConnectionsKernel<256, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 128:
				SumConnectionsKernel<128, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 64:
				SumConnectionsKernel< 64, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 32:
				SumConnectionsKernel< 32, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case 16:
				SumConnectionsKernel< 16, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  8:
				SumConnectionsKernel<  8, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  4:
				SumConnectionsKernel<  4, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  2:
				SumConnectionsKernel<  2, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
			case  1:
				SumConnectionsKernel<  1, SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		}
	}
	checkCUDAError("cuda_inputCalculation2");
}
