#include "cudaNeuralNet.h"

CudaNeuralNet::CudaNeuralNet():NeuralNet()
{
	host_inputSizes = NULL;
	host_inputs = NULL;
	host_types = NULL;
	dev_inputs = NULL;
	inDevice = 0;
}

CudaNeuralNet::~CudaNeuralNet()
{
	//TODO descomentar cuando ya no exista freeNeuralNet()
	/*
	freeDevice();
	delete[] host_inputs;
	delete[] host_inputSizes;
	delete[] host_types;*/
}

void CudaNeuralNet::freeNeuralNet()
{
	freeDevice();
	delete[] host_inputs;
	delete[] host_inputSizes;
	delete[] host_types;

	if (layers) {
		for (unsigned i=0; i < numberLayers; i++) {
			layers[i]->freeLayer();
			delete (layers[i]);
		}
		free(layers);
	}
	if (layerConnectionsGraph) {
		free(layerConnectionsGraph);
	}
	if (inputs) {
		free(inputs);
	}
	if (inputsToLayersGraph){
		free(inputsToLayersGraph);
	}
	if (outputs) {
		free(outputs);
	}
	if (outputLayers) {
		free(outputLayers);
	}
}

Layer* CudaNeuralNet::newLayer()
{
	return new CudaLayer();
}

Layer* CudaNeuralNet::newLayer(VectorType inputType, VectorType outputType, FunctionType functionType)
{
	return new CudaLayer(inputType, outputType, functionType);
}

void CudaNeuralNet::hostToDevice(){

	host_inputs = new void*[numberInputs];
	host_inputSizes = new unsigned[numberInputs];
	host_types = new VectorType[numberInputs];

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
		d_inputs = new void*[numInputsCurrentLayer];

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
		delete[] d_inputs;
	}
	inDevice = 1;
}

void CudaNeuralNet::freeDevice()
{
	if (dev_inputs != NULL){
		FreeInputs(dev_inputs, numberInputs);
		dev_inputs = NULL;
	}
	for (unsigned i=0; i<numberLayers; i++){
		((CudaLayer*)layers[i])->freeDevice();
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

