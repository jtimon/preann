#include "xmmLayer.h"

XmmLayer::XmmLayer()
{
}

XmmLayer::~XmmLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			delete(connections[i]);
		}
		mi_free(inputs);
		mi_free(connections);
		inputs = NULL;
		connections = NULL;
	}
	if (thresholds) {
		delete(thresholds);
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

