#include "xmmLayer.h"

XmmLayer::XmmLayer()
{
}

void XmmLayer::init(unsigned size, VectorType outputType, FunctionType functionType)
{
	this->functionType = functionType;
	output = newVector(size, outputType);
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

	unsigned numLoops;
	unsigned weighPos = 0;

	if (input->getVectorType() == FLOAT) {

		numLoops = ((input->getSize()-1)/FLOATS_PER_BLOCK)+1;

		for (unsigned j=0; j < output->getSize(); j++){

			float auxResult;
			XMMreal(inputPtr, numLoops,
					(((float*)inputWeighs) + weighPos), auxResult);
			results[j] += auxResult;
			weighPos += input->getSize();
		}
	}
	else {
		numLoops = ((input->getSize()-1)/BYTES_PER_BLOCK)+1;

		if (input->getVectorType() == BIT) {
			for (unsigned j=0; j < output->getSize(); j++){

				//printf("weighsOffsetPerOutput %d inputSize %d outputSize %d loops %d weighPos %d \n", weighsOffsetPerOutput, input->getSize(), output->getSize(), numLoops, weighPos);
				results[j] += XMMbinario(inputPtr, numLoops,
						(((unsigned char*)inputWeighs) + weighPos));
				weighPos += input->getSize();
			}
		}
		else if (input->getVectorType() == SIGN) {
			for (unsigned j=0; j < output->getSize(); j++){

				results[j] += XMMbipolar(inputPtr, numLoops,
									(((unsigned char*)inputWeighs) + weighPos));
				weighPos += input->getSize();
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

//void XmmLayer::saveWeighs(FILE* stream)
//{
//	if (inputType == FLOAT) {
//
//	} else {
//
//	}
//	byteSize *= output->getSize();
//	void* data = mi_malloc(byteSize);
//
//	fwrite(thresholds, output->getSize() * sizeof(float), 1, stream);
//	for (unsigned i=0; i < numberInputs; i++){
//		unsigned size;
//		if (inputs[i]->getVectorType() == FLOAT){
//			unsigned numBlocks = ((inputs[i]->getSize() -1)/FLOATS_PER_BLOCK) + 1;
//			size = output->getSize() * numBlocks * FLOATS_PER_BLOCK * sizeof(float);
//		} else {
//			unsigned numBlocks = ((inputs[i]->getSize() -1)/BYTES_PER_BLOCK) + 1;
//			size = output->getSize() * inputs[i]->getSize() * sizeof(unsigned char);
//		}
//		fwrite(weighs[i], size, 1, stream);
//	}
//}
//
//void XmmLayer::loadWeighs(FILE* stream)
//{
//	fread(thresholds, output->getSize() * sizeof(float), 1, stream);
//	for (unsigned i=0; i < numberInputs; i++){
//		unsigned size;
//		if (inputs[i]->getVectorType() == FLOAT){
//			size = output->getSize() * inputs[i]->getSize() * sizeof(float);
//		} else {
//			size = output->getSize() * inputs[i]->getSize() * sizeof(unsigned char);
//		}
//		fread(weighs[i], size, 1, stream);
//	}
//}

void* XmmLayer::newWeighs(unsigned inputSize, VectorType inputType)
{
	unsigned size;
	if (inputType == FLOAT) {
		size = output->getSize() * inputSize * sizeof(float);
	} else {
		size = output->getSize() * inputSize * sizeof(unsigned char);
	}
	//make sure that the xmm code has enough memory to read in the last loop
	size = (((size -1)/BYTES_PER_BLOCK) + 1) * BYTES_PER_BLOCK;
	void* toReturn = mi_malloc(size);
	//put to zero all the weighs reserved so they won't be counted when the sse2 algorithm reads them
	if (inputType == FLOAT) {
		for (unsigned i=0; i < size/sizeof(float); i++) {
			((float*)toReturn)[i] = 0;
		}
	} else {
		for (unsigned i=0; i < size; i++) {
			((unsigned char*)toReturn)[i] = 128;
		}
	}

	return toReturn;
}

