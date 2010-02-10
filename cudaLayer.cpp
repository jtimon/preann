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

	struct_Layer* layerAux = (struct_Layer*) mi_malloc(sizeof(struct_Layer));

	layerAux->functionType = this->functionType;
	layerAux->numberInputLayers = this->numberInputs;
	layerAux->totalWeighsPerOutput = this->totalWeighsPerOutput;

	layerAux->weighs = this->weighs;
	layerAux->thresholds = this->thresholds;

	layerAux->outputSize = this->output->getSize();
	layerAux->outputNeurons = this->output->getDataPointer();

	unsigned* inputLayerSize = (unsigned*) mi_malloc(sizeof(unsigned) * this->numberInputs);
	void** inputNeurons = (void**) mi_malloc(sizeof(void*) * this->numberInputs);

	for(unsigned i=0; i < this->numberInputs; i++){

		inputLayerSize[i] = this->getInput(i)->getSize();
		inputNeurons[i] = this->getInput(i)->getDataPointer();
	}

	layerAux->inputLayerSize = inputLayerSize;
	layerAux->inputNeurons = inputNeurons;

	deviceLayer = LayerHostToDevice(layerAux, inputType, outputType);

	mi_free(layerAux);
	mi_free(inputLayerSize);
	mi_free(inputNeurons);
}

void CudaLayer::setInputsInDevice(void **inputs)
{
	SetInputsInDevice(deviceLayer, inputs);
}

float* CudaLayer::getDeviceOutput()
{
	if (!deviceLayer) {
		string error = "Cannot get the device output pointer of a CudaLayer that is not in device memory.";
		throw error;
	}
	return (float*)deviceLayer->outputNeurons;
}

void CudaLayer::freeDevice()
{
	if (deviceLayer){
		FreeDevice(deviceLayer);
		deviceLayer = NULL;
	}
}

void CudaLayer::outputToHost()
{
	if (!output) {
		string error = "Cannot calculate bring the output from the device memory of a Layer without output.";
		throw error;
	}
	OutputToHost(output->getDataPointer(), deviceLayer, outputType);
}

void CudaLayer::calculateOutput(){
	
	if (!deviceLayer) {
		string error = "Cannot calculate the output of a CudaLayer that is not in device memory.";
		throw error;
	}
	LayerCalculation(deviceLayer, THREADS_PER_BLOCK, inputType, outputType);
}


