#include "cuda_code.h"

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Cuda error: %s : %s.\n", msg, cudaGetErrorString(err));
		exit( EXIT_FAILURE);
	}
}

/// ACTIVATION
__device__
float Func(float number, FunctionType functionType)
{
	switch (functionType) {

	case FT_BINARY_STEP:
		if (number > 0) {
			return 1;
		} else {
			return 0;
		}
	case FT_BIPOLAR_STEP:
		if (number > 0) {
			return 1;
		} else {
			return -1;
		}
	case SIGMOID:
		return 1.0f / (1.0f - exp(-number));
	case FT_BIPOLAR_SIGMOID:
		return -1.0f + (2.0f / (1.0f + exp(-number)));
	case FT_HYPERBOLIC_TANGENT:
		return tanh(number);
	case FT_IDENTITY:
	default:
		return number;
	}
}

__global__
void activation_float_kernel(float* results, float* output, unsigned output_sz,
		FunctionType functionType)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < output_sz)
		output[idx] = Func(results[idx], functionType);
}

__global__
void activation_bit_kernel(float* results, unsigned* output, unsigned output_sz)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned offset = idx * BITS_PER_UNSIGNED;

	if (output_sz > offset) {

		unsigned toRead = min(BITS_PER_UNSIGNED, output_sz - offset);
		unsigned threadOutput = 0;
		unsigned mask = 0x80000000;

		for (unsigned i = 0; i < toRead; i++) {
			if (results[offset + i] > 0) {
				threadOutput |= mask;
			} else {
				threadOutput &= ~mask;
			}
			mask >>= 1;
		}
		output[idx] = threadOutput;
	}
}

extern "C" void cuda_activation(void* data, unsigned size, BufferType bufferType, float* results, FunctionType functionType, unsigned block_size)
{
	unsigned grid_size;

	switch (bufferType){
	case BT_BYTE:
		{
			std::string error = "cuda_activation is not implemented for BufferType BYTE.";
			throw error;
		}
	case BT_FLOAT:
		{
			grid_size = ((size - 1) / block_size) + 1;
			activation_float_kernel<<< grid_size, block_size >>>(results, (float*)data, size, functionType);
		}
		break;
	case BT_BIT:
	case BT_SIGN:
		{
			grid_size = ((size - 1) / (block_size * BITS_PER_UNSIGNED)) + 1;
			activation_bit_kernel<<< grid_size, block_size >>>(results, (unsigned*)data, size);
		}
		break;
	}
	checkCUDAError("activation");
}

// MEMORY MANAGEMENT

extern "C" void* cuda_malloc(unsigned byteSize)
{
	void* ptr;
	cudaMalloc((void**) &(ptr), byteSize);

	checkCUDAError("malloc");
	return ptr;
}

extern "C" void cuda_free(void* d_ptr)
{
	cudaFree(d_ptr);
	checkCUDAError("free");
}

extern "C" void cuda_copyToDevice(void* d_dest, void* h_src, unsigned count)
{
	cudaMemcpy(d_dest, h_src, count, cudaMemcpyHostToDevice);
	checkCUDAError("copyToDevice");
}

extern "C" void cuda_copyToHost(void* h_dest, void* d_src, unsigned count)
{
	cudaMemcpy(h_dest, d_src, count, cudaMemcpyDeviceToHost);
	checkCUDAError("copyToHost");
}

// INITIALIZATION

template <class bufferType>
__global__
void SetValueToAnArrayKernel(bufferType* data, unsigned size, bufferType value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		data[idx] = value;
}

extern "C" void cuda_setZero(void* data, unsigned byteSize, BufferType bufferType, unsigned block_size)
{
	unsigned grid_size;
	unsigned size;

	switch (bufferType){
	case BT_BYTE:
		size = byteSize / sizeof(unsigned char);
		grid_size = ((size - 1) / block_size) + 1;
		SetValueToAnArrayKernel<unsigned char><<< grid_size, block_size >>>((unsigned char*)data, size, (unsigned char)0);
		break;
	case BT_FLOAT:
		size = byteSize / sizeof(float);
		grid_size = ((size - 1) / block_size) + 1;
		SetValueToAnArrayKernel<float><<< grid_size, block_size >>>((float*)data, size, 0);
		break;
	case BT_BIT:
	case BT_SIGN:
		cudaMemset(data, 0, byteSize);
		break;
	}
}

// GENETIC OPERATORS

template <class type>
__global__
void crossoverKernel(type* buffer1, type* buffer2, unsigned* bitBuffer, unsigned size)
{
	unsigned weighPos = (blockIdx.x * blockDim.x * BITS_PER_UNSIGNED) + threadIdx.x;
	unsigned maxPosForThisBlock = min ( (blockIdx.x + 1) * blockDim.x * BITS_PER_UNSIGNED,
										size);
	unsigned bitsForTheThread, mask;
	if (weighPos < maxPosForThisBlock) {
		bitsForTheThread = bitBuffer[(blockIdx.x * blockDim.x) + threadIdx.x];
		mask = 0x80000000;
	}
	__syncthreads();
	while (weighPos < maxPosForThisBlock){
		if (mask & bitsForTheThread){
			type aux = buffer1[weighPos];
			buffer1[weighPos] = buffer2[weighPos];
			buffer2[weighPos] = aux;
		}
		weighPos += blockDim.x;
		mask >>= 1;
	}
}

extern "C"
void cuda_crossover(void* buffer1, void* buffer2, unsigned* bitBuffer, unsigned size, BufferType bufferType,unsigned block_size)
{
	unsigned grid_size = ((size - 1)/(block_size * BITS_PER_UNSIGNED)) + 1;

	switch (bufferType){
        case BT_BYTE:
		crossoverKernel<unsigned char><<< grid_size, block_size >>>
				((unsigned char*)buffer1, (unsigned char*)buffer2, (unsigned*)bitBuffer, size);

        break;
    case BT_FLOAT:
    	crossoverKernel<float><<< grid_size, block_size >>>
				((float*)buffer1, (float*)buffer2, (unsigned*)bitBuffer, size);
		break;
	case BT_BIT:
	case BT_SIGN:
		{
		std::string error = "cuda_crossover is not implemented for BufferType BIT nor SIGN.";
		throw error;
		}
	}
}

//TODO CU es necesario usar un kernel para esto ??
__global__
void resetFloatKernel(float* buffer, unsigned pos)
{
	if (threadIdx.x == 0){
		buffer[pos] = 0;
	}
}

__global__
void resetByteKernel(unsigned char* buffer, unsigned pos)
{
	if (threadIdx.x == 0){
		buffer[pos] = 128;
	}
}

__global__
void mutateFloatKernel(float* buffer, unsigned pos, float mutation)
{
	if (threadIdx.x == 0){
		buffer[pos] += mutation;
	}
}

__global__
void mutateByteKernel(unsigned char* buffer, unsigned pos, int mutation)
{
	if (threadIdx.x == 0){
		int result = mutation + buffer[pos];
		if (result <= 0){
			buffer[pos] = 0;
		}
		else if (result >= 255) {
			buffer[pos] = 255;
		}
		else {
			buffer[pos] = (unsigned char)result;
		}
	}
}

extern "C" void cuda_mutate(void* buffer, unsigned pos, float mutation, BufferType bufferType)
{
	switch (bufferType){
	case BT_BYTE:
		mutateByteKernel<<< 1, 8 >>>((unsigned char*)buffer, pos, (int)mutation);
		break;
	case BT_FLOAT:
		mutateFloatKernel<<< 1, 8 >>>((float*)buffer, pos, mutation);
		break;
	case BT_BIT:
	case BT_SIGN:
		{
		std::string error = "cuda_mutate is not implemented for BufferType BIT nor SIGN.";
		throw error;
		}
	}
}

extern "C" void cuda_reset(void* buffer, unsigned pos, BufferType bufferType)
{
	switch (bufferType){
	case BT_BYTE:
		resetByteKernel<<< 1, 8 >>>((unsigned char*)buffer, pos);
		break;
	case BT_FLOAT:
		resetFloatKernel<<< 1, 8 >>>((float*)buffer, pos);
		break;
	case BT_BIT:
	case BT_SIGN:
		{
		std::string error = "cuda_reset is not implemented for BufferType BIT nor SIGN.";
		throw error;
		}
	}
}

// CL_LAYER CALCULATION

__global__
void SumFloatsConnectionsKernel(float* inputs, unsigned input_size, unsigned output_size, float* weighs, float* results)
{
	extern __shared__ float sdata[];

	unsigned outputNeuron = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned weighsOffset = outputNeuron * input_size;
	float result = 0;

	unsigned pos = threadIdx.x;
	while (pos < input_size) {

		sdata[pos] = inputs[pos];
		pos += blockDim.x;
	}
	__syncthreads();

	if (outputNeuron < output_size) {

		//////////////////////////
		for (unsigned i = 0; i < input_size; i++) {
			result += sdata[i] * weighs[weighsOffset + i];
			//printf(" peso %f ", weighs[weighsOffset + i]);
		}
		/////TODO TR OTRA OPCION
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

template <BufferType inputType>
__global__
void SumBitsConnectionsKernel(unsigned* inputs, unsigned input_size, unsigned output_size, unsigned char* weighs, float* results)
{
	extern __shared__ unsigned shared_inputs[];

	unsigned tid = threadIdx.x;
	unsigned input_blocks_to_read = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
	unsigned readingLoops = ((input_blocks_to_read - 1) / blockDim.x) + 1;

	unsigned pos = tid;

	for (unsigned i=0; i < readingLoops; i++) {
		if (pos < input_blocks_to_read) {
			shared_inputs[pos] = inputs[pos];
		}
		pos += blockDim.x;
	}
	__syncthreads();

	unsigned outputNeuron = blockIdx.x*blockDim.x + threadIdx.x;
	if (outputNeuron < output_size) {

		float result = 0;
		unsigned weighsOffset = (outputNeuron * input_size);

		for (unsigned i=0; i < input_blocks_to_read; i++) {

			//TODO TCC check performance penalty (this is just for BT_SIGN)
			unsigned maxBits = min(BITS_PER_UNSIGNED, input_size - (i * BITS_PER_UNSIGNED));

			unsigned input_block = shared_inputs[i];
			unsigned mask = 0x80000000;
			for (unsigned j=0; j < maxBits; j++) {

				if (input_block & mask) {
					result += weighs[weighsOffset] - 128;
				} else {
					if (inputType == BT_SIGN) {
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

__global__
void SumFloatsInvertedConnectionsKernel(float* inputs, unsigned input_size,
		float* weighs, float* results, unsigned output_size)
{
	extern __shared__ float sdata[];

	unsigned input_pos = threadIdx.x;
	while (input_pos < input_size) {

		sdata[input_pos] = inputs[input_pos];
		input_pos += blockDim.x;
	}
	__syncthreads();

	unsigned output_pos = blockIdx.x * blockDim.x + threadIdx.x;
	float result = 0;

	if (output_pos < output_size) {

		for (unsigned i = 0; i < input_size; i++) {
			result += sdata[i] * weighs[output_pos + (i * output_size)];
		}
		results[output_pos] += result;
	}
}

template <BufferType inputType>
__global__
void SumBitsInvertedConnectionsKernel(unsigned* inputs, unsigned input_size, unsigned output_size, unsigned char* weighs, float* results)
{
	extern __shared__ unsigned shared_inputs[];

	unsigned tid = threadIdx.x;
	unsigned input_blocks_to_read = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
	unsigned readingLoops = ((input_blocks_to_read - 1) / blockDim.x) + 1;

	unsigned pos = tid;

	for (unsigned i=0; i < readingLoops; i++) {
		if (pos < input_blocks_to_read) {
			shared_inputs[pos] = inputs[pos];
		}
		pos += blockDim.x;
	}
	__syncthreads();

	unsigned outputNeuron = blockIdx.x*blockDim.x + threadIdx.x;
	if (outputNeuron < output_size) {

		float result = 0;

		for (unsigned i=0; i < input_blocks_to_read; i++) {

			//TODO TCC check performance penalty (this is just for BT_SIGN)
			unsigned maxBits = min(BITS_PER_UNSIGNED, input_size - (i * BITS_PER_UNSIGNED));

			unsigned weighsOffset = (i * BITS_PER_UNSIGNED * output_size) + outputNeuron;
			unsigned input_block = shared_inputs[i];
			unsigned mask = 0x80000000;
			for (unsigned j=0; j < maxBits; j++) {

				if (input_block & mask) {
					result += weighs[weighsOffset] - 128;
				} else {
					if (inputType == BT_SIGN) {
						result += 128 - weighs[weighsOffset];
					}
				}
				weighsOffset += output_size;
				mask >>= 1;
			}
		}
		results[outputNeuron] += result;
	}
}

extern "C" void cuda_inputCalculation(void* inputPtr, unsigned input_size,
		BufferType inputType, unsigned output_size, void* weighs,
		float* results, unsigned block_size)
{
	unsigned grid_size = ((output_size - 1) / block_size) + 1;
	unsigned shared_mem_size;

	if (inputType == BT_BYTE) {
		std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
		throw error;
	}
	else if (inputType == BT_FLOAT) {
		if (input_size > 4032) {
			string error = "The maximum float input size is 4032.";
			throw error;
		}
		shared_mem_size = input_size * sizeof(float);

		SumFloatsConnectionsKernel<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtr, input_size, output_size, (float*)weighs, results);
	} else {

		shared_mem_size =(((input_size - 1)/BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
		if (shared_mem_size > 16128) {
			//16128 * 8
			string error = "The maximum bit/sign input size is 129024.";
			throw error;
		}
		if (inputType == BT_BIT) {
			SumBitsConnectionsKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		} else {
			SumBitsConnectionsKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		}
	}
}

extern "C" void cuda_inputCalculationInvertedMatrix(void* inputPtr, unsigned input_size,
		BufferType inputType, unsigned output_size, void* weighs,
		float* results, unsigned block_size)
{
	unsigned grid_size = ((output_size - 1) / block_size) + 1;
	unsigned shared_mem_size;

	if (inputType == BT_BYTE) {
		std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
		throw error;
	}
	else if (inputType == BT_FLOAT) {
		while (input_size > CUDA_MAX_SHARED_FLOATS) {

			shared_mem_size = CUDA_MAX_SHARED_FLOATS * sizeof(float);
			SumFloatsInvertedConnectionsKernel<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtr, CUDA_MAX_SHARED_FLOATS, (float*)weighs, results, output_size);
			inputPtr = (void*) ((float*) inputPtr + CUDA_MAX_SHARED_FLOATS);
			weighs = (void*) ((float*) weighs + (CUDA_MAX_SHARED_FLOATS
					* output_size));
			input_size -= CUDA_MAX_SHARED_FLOATS;
		}
		shared_mem_size = input_size * sizeof(float);
		SumFloatsInvertedConnectionsKernel<<< grid_size, block_size, shared_mem_size >>>((float*)inputPtr, input_size, (float*)weighs, results, output_size);
	} else {
		//TODO TCC esta parte no funciona bien
		while (input_size > CUDA_MAX_SHARED_BITS) {

			shared_mem_size = CUDA_MAX_SHARED_FLOATS * sizeof(unsigned);
			// TODO TCC probar sin emulación
//			printf("grid_size %d, block_size %d, shared_mem_size %d \n", grid_size, block_size, shared_mem_size);
			if (inputType == BT_BIT) {
				SumBitsInvertedConnectionsKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, CUDA_MAX_SHARED_BITS, output_size, (unsigned char*)weighs, results);
			} else {
				SumBitsInvertedConnectionsKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, CUDA_MAX_SHARED_BITS, output_size, (unsigned char*)weighs, results);
			}
			inputPtr = (void*)((float*)inputPtr + CUDA_MAX_SHARED_FLOATS);
			weighs = (void*)((float*)weighs + (CUDA_MAX_SHARED_BITS * output_size));
			input_size -= CUDA_MAX_SHARED_BITS;
		}
		shared_mem_size =(((input_size - 1)/BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
		// TODO TCC probar sin emulación
		//printf("grid_size %d, block_size %d, shared_mem_size %d \n", grid_size, block_size, shared_mem_size);
		if (inputType == BT_BIT) {
			SumBitsInvertedConnectionsKernel<BT_BIT><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		} else {
			SumBitsInvertedConnectionsKernel<BT_SIGN><<< grid_size, block_size, shared_mem_size >>>((unsigned*)inputPtr, input_size, output_size, (unsigned char*)weighs, results);
		}
	}
}

template <unsigned int blockSize, BufferType inputType>
__global__
void SumConnectionsKernel(void* inputPtr, unsigned input_size, unsigned output_size, void* weighs, float* results)
{
	extern __shared__ float sdata[];

	unsigned weighsOffset = (blockIdx.x * input_size);

	float result = 0;
	unsigned i = threadIdx.x;

	if (inputType == BT_FLOAT) {
		while (i < input_size) {
			result += ((float*)inputPtr)[i] * ((float*)weighs)[weighsOffset + i];
			i += blockDim.x;
		}
	} else {
		weighsOffset += threadIdx.x * BITS_PER_UNSIGNED;

		unsigned input_blocks_to_read = ((input_size - 1) / BITS_PER_UNSIGNED) + 1;
		while (i < input_blocks_to_read) {

			//TODO TCC check performance penalty (this is just for BT_SIGN)
			unsigned maxBits = min(BITS_PER_UNSIGNED, input_size - (i * BITS_PER_UNSIGNED));

			unsigned mask = 0x80000000;
			unsigned currentInput = ((unsigned*)inputPtr)[i];

			for (unsigned j=0; j < maxBits; j++) {

				if (currentInput & mask) {
					result += ((unsigned char*)weighs)[weighsOffset + j] - 128;
				} else {
					if (inputType == BT_SIGN) {
						result -= ((unsigned char*)weighs)[weighsOffset + j] - 128;
					}
				}
				mask >>= 1;
			}
			i += blockSize;
			weighsOffset += blockDim.x * BITS_PER_UNSIGNED;
		}
	}

	unsigned tid = threadIdx.x;
	sdata[tid] = result;
	__syncthreads();

	if (blockSize >= 512) {if (tid < 256) {sdata[tid] += sdata[tid + 256];}__syncthreads();}
	if (blockSize >= 256) {if (tid < 128) {sdata[tid] += sdata[tid + 128];}__syncthreads();}
	if (blockSize >= 128) {if (tid < 64) {sdata[tid] += sdata[tid + 64];}__syncthreads();}

#if __DEVICE_EMULATION__
	if (blockSize >= 64) {if (tid < 32) {sdata[tid] += sdata[tid + 32];}__syncthreads();}
	if (blockSize >= 32) {if (tid < 16) {sdata[tid] += sdata[tid + 16];}__syncthreads();}
	if (blockSize >= 16) {if (tid < 8) {sdata[tid] += sdata[tid + 8];}__syncthreads();}
	if (blockSize >= 8) {if (tid < 4) {sdata[tid] += sdata[tid + 4];}__syncthreads();}
	if (blockSize >= 4) {if (tid < 2) {sdata[tid] += sdata[tid + 2];}__syncthreads();}
	if (blockSize >= 2) {if (tid < 1) {sdata[tid] += sdata[tid + 1];}__syncthreads();}
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
	if (tid == 0) {
		results[blockIdx.x] += sdata[0];
	}
}

extern "C" void cuda_inputCalculationReduction(void* inputPtr, unsigned input_size, BufferType inputType, unsigned output_size, void* weighs,
		float* results, unsigned block_size)
{
	unsigned grid_size = output_size;
	unsigned shared_mem_size = block_size * sizeof(float);

	if (inputType == BT_BYTE) {
		std::string error = "cuda_inputCalculation is not implemented for BufferType BYTE as input.";
		throw error;
	}
	else if (inputType == BT_FLOAT) {
		switch (block_size) {
		case 512:
			SumConnectionsKernel<512, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 256:
			SumConnectionsKernel<256, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 128:
			SumConnectionsKernel<128, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 64:
			SumConnectionsKernel< 64, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 32:
			SumConnectionsKernel< 32, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 16:
			SumConnectionsKernel< 16, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 8:
			SumConnectionsKernel< 8, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 4:
			SumConnectionsKernel< 4, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 2:
			SumConnectionsKernel< 2, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 1:
			SumConnectionsKernel< 1, BT_FLOAT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		}
	} else if (inputType == BT_BIT) {
		switch (block_size) {
		case 512:
			SumConnectionsKernel<512, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 256:
			SumConnectionsKernel<256, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 128:
			SumConnectionsKernel<128, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 64:
			SumConnectionsKernel< 64, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 32:
			SumConnectionsKernel< 32, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 16:
			SumConnectionsKernel< 16, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 8:
			SumConnectionsKernel< 8, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 4:
			SumConnectionsKernel< 4, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 2:
			SumConnectionsKernel< 2, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 1:
			SumConnectionsKernel< 1, BT_BIT><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		}
	} else {
		switch (block_size) {
		case 512:
			SumConnectionsKernel<512, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 256:
			SumConnectionsKernel<256, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 128:
			SumConnectionsKernel<128, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 64:
			SumConnectionsKernel< 64, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 32:
			SumConnectionsKernel< 32, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 16:
			SumConnectionsKernel< 16, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 8:
			SumConnectionsKernel< 8, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 4:
			SumConnectionsKernel< 4, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 2:
			SumConnectionsKernel< 2, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		case 1:
			SumConnectionsKernel< 1, BT_SIGN><<< grid_size, block_size, shared_mem_size >>>(inputPtr, input_size, output_size, weighs, results); break;
		}
	}
	checkCUDAError("cuda_inputCalculation2");
}

