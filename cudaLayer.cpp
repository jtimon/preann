#include "cudaLayer.h"

unsigned CudaLayer::block_size = 128;
unsigned CudaLayer::version = 0;
/*
CudaLayer::CudaLayer()
{
	deviceLayer = NULL;
}*/

CudaLayer::CudaLayer(VectorType inputType, VectorType outputType, FunctionType functionType): Layer(inputType, outputType, functionType)
{
	deviceLayer = NULL;
}

CudaLayer::~CudaLayer()
{
	freeDevice();

	if (inputs) {
		mi_free(inputs);
	}
	if (thresholds) {
		mi_free(thresholds);
	}
	if (weighs) {
		mi_free(weighs);
	}
	if (output) {
		delete (output);
	}
}

void CudaLayer::saveWeighs(FILE *stream)
{
	fwrite(thresholds, output->getSize() * sizeof(float), 1, stream);
	unsigned size;
	if (inputType == FLOAT){
		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
	} else {
		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
	}
	fwrite(weighs, size, 1, stream);
}

void CudaLayer::loadWeighs(FILE *stream)
{
	fread(thresholds, output->getSize() * sizeof(float), 1, stream);
	unsigned size;
	if (inputType == FLOAT){
		size = output->getSize() * totalWeighsPerOutput * sizeof(float);
	} else {
		size = output->getSize() * totalWeighsPerOutput * sizeof(unsigned char);
	}
	fread(weighs, size, 1, stream);
}

void CudaLayer::setSizes(unsigned  totalWeighsPerOutput, unsigned  outputSize)
{
	if (!output) {
		output = new Vector(outputSize, outputType);
		thresholds = (float*) mi_malloc(sizeof(float) * outputSize);
	} else if (output->getSize() != outputSize) {

		cout<<"Warning: a layer is changing the location of its output."<<endl;
		delete (output);
		if (thresholds) {
			mi_free(thresholds);
		}
		output = new Vector(outputSize, outputType);
		thresholds = (float*)mi_malloc(sizeof(float) * outputSize);
	}
	if (totalWeighsPerOutput > 0){
		if (inputType == FLOAT){

			weighs = mi_malloc(sizeof(float) * outputSize * totalWeighsPerOutput);
			for (unsigned i=0; i < outputSize * totalWeighsPerOutput; i++){
				((float*)weighs)[i] = 0;
			}
		} else {
			weighs = mi_malloc(sizeof(unsigned char) * outputSize * totalWeighsPerOutput);
			for (unsigned i=0; i < outputSize * totalWeighsPerOutput; i++){
				((unsigned char*)weighs)[i] = 128;
			}
		}
	}
	this->totalWeighsPerOutput = totalWeighsPerOutput;
}

Layer* CudaLayer::newCopy()
{
	Layer* copy = new CudaLayer(inputType, outputType, functionType);

	copy->setSizes(totalWeighsPerOutput, output->getSize());

	return copy;
}

void CudaLayer::toDevice(){

	if (deviceLayer != NULL) {
		FreeDevice(deviceLayer);
	}

	struct_Layer* layerAux = (struct_Layer*) mi_malloc(sizeof(struct_Layer));

	layerAux->h_functionType = this->functionType;
	layerAux->h_numberInputLayers = this->numberInputs;
	layerAux->h_totalWeighsPerOutput = this->totalWeighsPerOutput;

	layerAux->weighs = this->weighs;
	layerAux->thresholds = this->thresholds;

	layerAux->h_outputSize = this->output->getSize();
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
	switch(CudaLayer::version){
		case 0:
			LayerCalculation(deviceLayer, CudaLayer::block_size, inputType, outputType);
			break;
		case 1:
			LayerCalculation2(deviceLayer, CudaLayer::block_size, inputType, outputType);
			break;
		case 2:
			LayerCalculation3(deviceLayer, CudaLayer::block_size, inputType, outputType);
			break;
	}
}


