#include "cudaLayer.h"

CudaLayer::CudaLayer()
{
	deviceLayer = NULL;
}

CudaLayer::CudaLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
	deviceLayer = NULL;
}

CudaLayer::~CudaLayer()
{
	freeDevice();
}

void CudaLayer::toDevice(){

	if (deviceLayer != NULL) {
		FreeDevice(deviceLayer);
	}

	struct_Layer* layerAux = new struct_Layer;

	layerAux->functionType = this->functionType;
	layerAux->numberInputLayers = this->numberInputs;
	layerAux->totalWeighsPerOutput = this->totalWeighsPerOutput;

	layerAux->weighs = this->weighs;
	layerAux->thresholds = this->thresholds;

	layerAux->outputSize = this->output->getSize();
	layerAux->outputNeurons = this->output->getDataPointer();

	unsigned* inputLayerSize = new unsigned[this->numberInputs];
	void** inputNeurons = new void*[this->numberInputs];

	for(unsigned i=0; i < this->numberInputs; i++){

		inputLayerSize[i] = this->inputs[i]->getSize();
		inputNeurons[i] = this->inputs[i]->getDataPointer();
	}

	layerAux->inputLayerSize = inputLayerSize;
	layerAux->inputNeurons = inputNeurons;

	deviceLayer = LayerHostToDevice(layerAux, inputType, outputType);

	delete layerAux;
}

void CudaLayer::setInputsInDevice(void **inputs)
{
	SetInputsInDevice(deviceLayer, inputs);
}

float* CudaLayer::getDeviceOutput()
{
	return (float*)deviceLayer->outputNeurons;
}

void CudaLayer::freeDevice()
{
	if (deviceLayer != NULL){
		FreeDevice(deviceLayer);
		delete(deviceLayer);
		deviceLayer = NULL;
	}
}

void CudaLayer::outputToHost()
{
	OutputToHost(output->getDataPointer(), deviceLayer, outputType);
}

void CudaLayer::calculateOutput(){
	
	if (deviceLayer != NULL) {
		LayerCalculation(deviceLayer, THREADS_PER_BLOCK, inputType, outputType);
	}
}


