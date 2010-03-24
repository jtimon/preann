#include "cudaNeuralNet.h"

CudaNeuralNet::CudaNeuralNet():NeuralNet(CUDA)
{
	host_inputSizes = NULL;
	host_inputs = NULL;
	host_types = NULL;
	dev_inputs = NULL;
	inDevice = 0;
}

CudaNeuralNet::~CudaNeuralNet()
{
	freeDevice();
	mi_free(host_inputs);
	mi_free(host_inputSizes);
	mi_free(host_types);
}

void CudaNeuralNet::hostToDevice(){

	host_inputs = (void**) mi_malloc(sizeof(void*) * numberInputs);
	host_inputSizes = (unsigned*) mi_malloc(sizeof(unsigned) * numberInputs);
	host_types = (VectorType*) mi_malloc(sizeof(VectorType) * numberInputs);

	for (unsigned i=0; i<numberInputs; i++){

		host_inputSizes[i] = inputs[i]->getSize();
		host_inputs[i] = inputs[i]->getDataPointer();
		host_types[i] = inputs[i]->getVectorType();
	}
	dev_inputs = InputsToDevice(host_inputs, host_inputSizes, host_types, numberInputs);

	for (unsigned i=0; i<numberLayers; i++){
		((CudaLayer*)layers[i])->toDevice();
	}

	void** d_inputs;

	for (unsigned i=0; i<numberLayers; i++){

		unsigned numInputsCurrentLayer = layers[i]->getNumberInputs();
		d_inputs = (void**) mi_malloc(sizeof(void*) * numInputsCurrentLayer);

		for (unsigned j=0; j < numInputsCurrentLayer; j++) {

			void* host_ptr = layers[i]->getInput(j)->getDataPointer();

			unsigned char founded = 0;
			for (unsigned k=0; k<numberInputs && !founded; k++){

				if (host_ptr == inputs[k]->getDataPointer()){
					d_inputs[j] = dev_inputs[k];
					++founded;
				}
			}
			for (unsigned k=0; k<numberLayers && !founded; k++){

				if (host_ptr == layers[k]->getOutput()->getDataPointer()){
					d_inputs[j] = ((CudaLayer*)layers[k])->getDeviceOutput();
					++founded;
				}
			}
		}
		((CudaLayer*)layers[i])->setInputsInDevice(d_inputs);
		mi_free(d_inputs);
	}
	inDevice = 1;
}

void CudaNeuralNet::freeDevice()
{
	if (dev_inputs){
		FreeInputs(dev_inputs, numberInputs);
	}
	inDevice = 0;
}

void CudaNeuralNet::calculateOutput()
{
	if(!inDevice) hostToDevice();

	RefreshDeviceInputs(dev_inputs, host_inputs, host_inputSizes, host_types, numberInputs);

	for (unsigned i=0; i < numberLayers; i++){
		layers[i]->calculateOutput();
	}

	for (unsigned i=0; i < numberOutputs; i++){
		((CudaLayer*)layers[outputLayers[i]])->outputToHost();
	}
}

