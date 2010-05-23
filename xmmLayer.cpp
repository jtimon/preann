#include "xmmLayer.h"

XmmLayer::XmmLayer(unsigned size, VectorType outputType, FunctionType functionType): CppLayer(outputType, functionType)
{
	output = new XmmVector(size, outputType);
	thresholds = (float*)mi_malloc(sizeof(float) * size);
}

XmmLayer::~XmmLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			mi_free(weighs[i]);
		}
		mi_free(inputs);
		mi_free(weighs);
		inputs = NULL;
		weighs = NULL;
	}
	if (thresholds) {
		mi_free(thresholds);
		thresholds = NULL;
	}
	if (output) {
		delete (output);
		output = NULL;
	}
}

void XmmLayer::inputCalculation(Vector* input, void* inputWeighs, float* results)
{
	void* inputPtr = input->getDataPointer();

	unsigned weighsOffsetPerOutput;
	unsigned numLoops;
	unsigned weighPos = 0;

	if (input->getVectorType() == FLOAT) {

		numLoops = ((input->getSize()-1)/FLOATS_PER_BLOCK)+1;
		weighsOffsetPerOutput = numLoops * FLOATS_PER_BLOCK;

		for (unsigned j=0; j < output->getSize(); j++){

			float auxResult;
			XMMreal(inputPtr, numLoops,
					(((float*)inputWeighs) + weighPos), auxResult);
			results[j] += auxResult;
			weighPos += weighsOffsetPerOutput;
		}
	}
	else {
		numLoops = ((input->getSize()-1)/BYTES_PER_BLOCK)+1;
		weighsOffsetPerOutput = numLoops * BYTES_PER_BLOCK;

		if (input->getVectorType() == BIT) {
			//TODO funciona bien con 1024 y 4096 pero no con 1025 ni 4097
			for (unsigned j=0; j < output->getSize(); j++){

				//printf("weighsOffsetPerOutput %d inputSize %d outputSize %d loops %d weighPos %d \n", weighsOffsetPerOutput, input->getSize(), output->getSize(), numLoops, weighPos);
				results[j] += XMMbinario(inputPtr, numLoops,
						(((unsigned char*)inputWeighs) + weighPos));
				weighPos += weighsOffsetPerOutput;
			}
		}
		else if (input->getVectorType() == SIGN) {
			for (unsigned j=0; j < output->getSize(); j++){

				results[j] += XMMbipolar(inputPtr, numLoops,
									(((unsigned char*)inputWeighs) + weighPos));
				weighPos += weighsOffsetPerOutput;
			}
		}
	}

/*
	for (unsigned j=0; j < output->getSize(); j++){
		unsigned weighPos = j * input->getWeighsSize();

		if (input->getVectorType() == FLOAT) {
			float auxResult;
			XMMreal(inputPtr, ((XmmVector*)input)->getNumLoops(),
					(((float*)inputWeighs) + weighPos), auxResult);
			results[j] += auxResult;
		}
		else if (input->getVectorType() == BIT) {
			results[j] += XMMbinario(inputPtr, ((XmmVector*)input)->getNumLoops(),
					(((unsigned char*)inputWeighs) + weighPos));
		}
		else if (input->getVectorType() == SIGN) {
			results[j] += XMMbipolar(inputPtr, ((XmmVector*)input)->getNumLoops(),
								(((unsigned char*)inputWeighs) + weighPos));
		}
	}*/
}

void* XmmLayer::newWeighs(unsigned inputSize, VectorType inputType)
{
	unsigned byteSize;

	if (inputType == FLOAT) {
		unsigned numBlocks = ((inputSize -1)/FLOATS_PER_BLOCK) + 1;
		byteSize = numBlocks * FLOATS_PER_BLOCK * sizeof(float);
	} else {
		unsigned numBlocks = ((inputSize -1)/BYTES_PER_BLOCK) + 1;
		byteSize = numBlocks * BYTES_PER_BLOCK * sizeof(unsigned char);
	}
	byteSize *= output->getSize();
	void* data = mi_malloc(byteSize);

	//TODO esto no deberia ser necesario porque se supone que los bits no usados del input vienen anulados
	if (inputType == FLOAT){

		unsigned floatSize = byteSize/sizeof(float);
		for (unsigned i=0; i< floatSize; i++){
			((float*)data)[i] = 0;
		}
	}
	else {

		for (unsigned i=0; i < byteSize; i++){
			((unsigned char*)data)[i] = 128;
		}
	}
	//

	return data;
}





