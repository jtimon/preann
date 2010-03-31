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

	if (input->getVectorType() == FLOAT) {
		for (unsigned j=0; j < output->getSize(); j++){

			unsigned weighPos = j * input->getWeighsSize();
			float auxResult;
			XMMreal(inputPtr, ((XmmVector*)input)->getNumLoops(),
					(((float*)inputWeighs) + weighPos), auxResult);
			results[j] += auxResult;
		}
	}
	else if (input->getVectorType() == BIT) {
		for (unsigned j=0; j < output->getSize(); j++){

			unsigned weighPos = j * input->getWeighsSize();
			results[j] += XMMbinario(inputPtr, ((XmmVector*)input)->getNumLoops(),
					(((unsigned char*)inputWeighs) + weighPos));
		}
	}
	else if (input->getVectorType() == SIGN) {
		for (unsigned j=0; j < output->getSize(); j++){

			unsigned weighPos = j * input->getWeighsSize();
			results[j] += XMMbipolar(inputPtr, ((XmmVector*)input)->getNumLoops(),
								(((unsigned char*)inputWeighs) + weighPos));
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
	//TODO adaptar para tamaños no multiples de los bloques
	unsigned size;
	if (inputType == FLOAT) {
		size = output->getSize() * inputSize * sizeof(float);
	} else {
		size = output->getSize() * inputSize * sizeof(unsigned char);
	}
	return mi_malloc(size);
}


Layer* XmmLayer::newCopy()
{
	std::string error = "newCopy is not implemented for XmmLayer.";
	throw error;
}
/*
void XmmLayer::calculateOutput()
{
	if (!output) {
		string error = "Cannot calculate the output of a Layer without output.";
		throw error;
	}

	 //TODO quitar getNumLoops (sacar el bucle de inputs del de salidas) y ¿permitir varios tipos de entrada?
	float* results = (float*) mi_malloc(output->getSize() * sizeof(float));
	for (unsigned j=0; j < output->getSize(); j++) {
		results[j] = -thresholds[j];
	}

	unsigned inputOffset = 0;
	for (unsigned i=0; i < numberInputs; i++){

		void* inputPtr = inputs[i]->getDataPointer();

		for (unsigned j=0; j < output->getSize(); j++){
			unsigned weighPos = j * inputs[i]->getWeighsSize();

				if (inputType == FLOAT) {
					float auxResult;
					XMMreal(inputPtr, ((XmmVector*)inputs[i])->getNumLoops(),
							(((float**)weighs)[i] + weighPos), auxResult);
					results[j] += auxResult;
				}
				if (inputType == BIT) {
					int auxResult = 0;
					XMMbinario(inputPtr, ((XmmVector*)inputs[i])->getNumLoops(),
							(((unsigned char**)weighs)[i] + weighPos), auxResult);
					results[j] += auxResult;
				}
				if (inputType == SIGN) {
					results[j] += XMMbipolar(inputPtr, ((XmmVector*)inputs[i])->getNumLoops(),
										(((unsigned char**)weighs)[i] + weighPos));
				}
		}
	}

//	printf("----------------\n", 1);
//	for (unsigned i=0; i < output->getSize(); i++){
//		printf("%f ", results[i]);
//	}
//	printf("\n----------------\n", 1);
	output->activation(results, functionType);
	mi_free(results);
}

*/



