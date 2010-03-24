#include "neuralNet.h"

NeuralNet::NeuralNet()
{
	this->implementationType = C;

	layers = NULL;
	layerConnectionsGraph = NULL;
	numberLayers = 0;

	inputs = NULL;
	inputsToLayersGraph = NULL;
	numberInputs = 0;

	outputs = NULL;
	outputLayers = NULL;
	numberOutputs = 0;
}

NeuralNet::NeuralNet(ImplementationType implementationType)
{
	this->implementationType = implementationType;

	layers = NULL;
	layerConnectionsGraph = NULL;
	numberLayers = 0;

	inputs = NULL;
	inputsToLayersGraph = NULL;
	numberInputs = 0;

	outputs = NULL;
	outputLayers = NULL;
	numberOutputs = 0;
}


NeuralNet::~NeuralNet()
{
	if (layers) {
		for (unsigned i=0; i < numberLayers; i++) {
			delete (layers[i]);
		}
		mi_free(layers);
	}
	if (layerConnectionsGraph) {
		mi_free(layerConnectionsGraph);
	}
	if (inputs) {
		mi_free(inputs);
	}
	if (inputsToLayersGraph){
		mi_free(inputsToLayersGraph);
	}
	if (outputs) {
		mi_free(outputs);
	}
	if (outputLayers) {
		mi_free(outputLayers);
	}
}

void NeuralNet::setInput(Vector *input)
{
	if (inputs){
		inputs[0] = input;
	}
}

void NeuralNet::resetConnections()
{
	//TODO resetConnections
}

void NeuralNet::calculateOutput()
{
	for (unsigned i=0; i < numberLayers; i++){
		//cout<<"calculando capa "<<i<<endl;//TODO quitar
		layers[i]->calculateOutput();
	}
}

void NeuralNet::addInput(Vector* input)
{
	unsigned newNumberInputs = numberInputs+1;

	Vector** newInputs = (Vector**) mi_malloc(sizeof(Vector*) * newNumberInputs);
	if (inputs) {
		memcpy(newInputs, inputs, numberInputs * sizeof(Vector*));
		mi_free(inputs);
	}
	inputs = newInputs;
	inputs[numberInputs] = input;

	if (numberLayers > 0){
		if (inputsToLayersGraph) {
			unsigned char* newInputsToLayersGraph = (unsigned char*) mi_malloc(sizeof(unsigned char) * newNumberInputs * numberLayers);
			for (unsigned i=0; i < numberInputs; i++){
				for (unsigned j=0; j < numberLayers; j++){
					newInputsToLayersGraph[(i*numberLayers) + j] = inputsToLayersGraph[(i*numberLayers) + j];
				}
			}
			for (unsigned j=0; j < numberLayers; j++){
				newInputsToLayersGraph[(numberInputs*numberLayers) + j] = 0;
			}
			mi_free(inputsToLayersGraph);
			inputsToLayersGraph = newInputsToLayersGraph;
		} else {
			inputsToLayersGraph = (unsigned char*) mi_malloc(sizeof(unsigned char) * newNumberInputs * numberLayers);
			for (unsigned i=0; i < newNumberInputs * numberLayers; i++) {
				inputsToLayersGraph[i] = 0;
			}
		}
	}
	++numberInputs;
}


void NeuralNet::addLayer(Layer* layer)
{
	unsigned newNumberLayers = numberLayers + 1;

	Layer** newLayers = (Layer**) mi_malloc(sizeof(Layer*) * newNumberLayers);
	if (layers) {
		memcpy(newLayers, layers, numberLayers * sizeof(Layer*));
		mi_free(layers);
	}
	layers = newLayers;

	if (numberInputs > 0) {
		if (inputsToLayersGraph) {
			unsigned char* newInputsToLayersGraph = (unsigned char*) mi_malloc(sizeof(unsigned char) * numberInputs * newNumberLayers);
			for (unsigned i=0; i < numberInputs; i++){
				for (unsigned j=0; j < numberLayers; j++){
					newInputsToLayersGraph[(i*numberLayers) + j] = inputsToLayersGraph[(i*numberLayers) + j];
				}
				newInputsToLayersGraph[(i*numberLayers) + numberLayers] = 0;
			}
			mi_free(inputsToLayersGraph);
			inputsToLayersGraph = newInputsToLayersGraph;
		}
		else {
			inputsToLayersGraph = (unsigned char*) mi_malloc(sizeof(unsigned char) * numberInputs * newNumberLayers);
			for (unsigned i=0; i < numberInputs * newNumberLayers; i++) {
				inputsToLayersGraph[i] = 0;
			}
		}
	}

	unsigned char* newLayerConnectionsGraph = (unsigned char*) mi_malloc(sizeof(unsigned char) * newNumberLayers * newNumberLayers);
	for (unsigned i=0; i < newNumberLayers; i++){
		for (unsigned j=0; j < newNumberLayers; j++){
			if (i == numberLayers || j == numberLayers){
				newLayerConnectionsGraph[(i * newNumberLayers) + j] = 0;
			} else {
				newLayerConnectionsGraph[(i * newNumberLayers) + j] = layerConnectionsGraph[(i * numberLayers) + j];
			}
		}
	}
	if (layerConnectionsGraph) {
		mi_free(layerConnectionsGraph);
	}
	layerConnectionsGraph = newLayerConnectionsGraph;

	layers[numberLayers++] = layer;
}

void NeuralNet::addLayer(unsigned  size, VectorType sourceType, VectorType destinationType, FunctionType functiontype)
{
	Layer* layer = Factory::newLayer(implementationType, sourceType, destinationType, functiontype);
	addLayer(layer);
	layer->setSize(size);
}

void NeuralNet::addLayer(unsigned  size, VectorType sourceType, VectorType destinationType)
{
	addLayer(size, sourceType, destinationType, IDENTITY);
}

void* NeuralNet::createInput(unsigned size, VectorType vectorType)
{

}

void* NeuralNet::getInput(unsigned pos)
{

}

VectorType NeuralNet::getInputType(unsigned pos)
{

}

unsigned NeuralNet::getNumInputs()
{

}

void NeuralNet::addOutput(unsigned layerPos)
{
	if (layerPos >= numberLayers){
		char buffer[100];
		sprintf(buffer, "Cannot set the Layer in position %d as an output: there are just %d Layers.", layerPos, numberLayers);
		string error = buffer;
		throw error;
	}

	Vector** newOutputs = (Vector**) mi_malloc(sizeof(Vector*) * (numberOutputs+1));
	if (outputs) {
		memcpy(newOutputs, outputs, numberOutputs * sizeof(Vector*));
		mi_free(outputs);
	}
	outputs = newOutputs;

	unsigned* newOutputLayers = (unsigned*) mi_malloc(sizeof(unsigned) * (numberOutputs+1));
	if (outputLayers) {
		memcpy(newOutputLayers, outputLayers, numberOutputs * sizeof(unsigned));
		mi_free(outputLayers);
	}
	outputLayers = newOutputLayers;

	outputLayers[numberOutputs] = layerPos;
	outputs[numberOutputs] = layers[layerPos]->getOutput();
	++numberOutputs;

}

void* NeuralNet::getOutput(unsigned outputPos)
{
	if (outputPos >= numberLayers){
		char buffer[100];
		sprintf(buffer, "Cannot access the output in position %d: there are just %d outputs.", outputPos, numberLayers);
		string error = buffer;
		throw error;
	}

	return outputs[outputPos]->getDataPointer();
}

VectorType NeuralNet::getOutputType(unsigned outputPos)
{
	if (outputPos >= numberLayers){
		char buffer[100];
		sprintf(buffer, "Cannot access the output in position %d: there are just %d outputs.", outputPos, numberLayers);
		string error = buffer;
		throw error;
	}
	return outputs[outputPos]->getVectorType();
}

unsigned NeuralNet::getNumOutputs()
{
	return numberOutputs;
}

void NeuralNet::randomWeighs(float range)
{
	for (unsigned i=0; i < numberLayers; i++){
		layers[i]->randomWeighs(range);
	}
}

void NeuralNet::addInputConnection(unsigned  sourceInputPos, unsigned  destinationLayerPos)
{
	if (!inputsToLayersGraph){
		string error = "The inputs To Layers Graph is not set yet.";
		throw error;
	}
	if (sourceInputPos >= numberInputs){
		char buffer[100];
		sprintf(buffer, "Cannot connect input in position %d: there are just %d inputs.", sourceInputPos, numberInputs);
		string error = buffer;
		throw error;
	}
	if (destinationLayerPos >= numberLayers){
		char buffer[100];
		sprintf(buffer, "Cannot connect an input with the Layer in position %d: there are just %d Layers.", destinationLayerPos, numberLayers);
		string error = buffer;
		throw error;
	}

	layers[destinationLayerPos]->addInput(inputs[sourceInputPos]);
	inputsToLayersGraph[(sourceInputPos * numberLayers) + destinationLayerPos] = 1;
}

void NeuralNet::addLayersConnection(unsigned  sourceLayerPos, unsigned  destinationLayerPos)
{
	if (sourceLayerPos >= numberLayers || destinationLayerPos >= numberLayers) {
		char buffer[100];
		sprintf(buffer, "Cannot connect Layer in position %d with Layer in position %d: there are just %d Layers.", sourceLayerPos, destinationLayerPos, numberLayers);
		string error = buffer;
		throw error;
	}
	if (sourceLayerPos == destinationLayerPos){
		char buffer[100];
		sprintf(buffer, "Cannot connect Layer in position %d with itself.", sourceLayerPos);
		string error = buffer;
		throw error;
	}

	layers[destinationLayerPos]->addInput(layers[sourceLayerPos]->getOutput());
	layerConnectionsGraph[(sourceLayerPos * numberLayers) + destinationLayerPos] = 1;
}

void NeuralNet::createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType)
{
	createFeedForwardNet(numLayers, sizeLayers, hiddenLayersType, IDENTITY, 0, 0);
}

void NeuralNet::createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype)
{
	createFeedForwardNet(numLayers, sizeLayers, hiddenLayersType, functiontype, 0, 0);
}

void NeuralNet::createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype, unsigned floatOutputSize, unsigned bitOutputSize)
{
	if (numberInputs == 0){
		string error = "Cannot create a network with no inputs.";
		throw error;
	}
	unsigned currentLayerIndex = 0;
	unsigned firstFloatIndex;
	unsigned firstBitIndex;
	unsigned firstSignIndex;
	Layer* firstFloatLayer = NULL;
	Layer* firstBitLayer = NULL;
	Layer* firstSignLayer = NULL;

	for (unsigned i=0; i<numberInputs; i++)	{
		switch (inputs[i]->getVectorType()) {
		default:
		case FLOAT:
			if (firstFloatLayer == NULL){
				firstFloatLayer = Factory::newLayer(implementationType, FLOAT, hiddenLayersType, functiontype);
				addLayer(firstFloatLayer);
				firstFloatIndex = currentLayerIndex++;
			}
			addInputConnection(i, firstFloatIndex);
			break;
		case BIT:
			if (firstBitLayer == NULL){
				firstBitLayer = Factory::newLayer(implementationType, BIT, hiddenLayersType, functiontype);
				addLayer(firstBitLayer);
				firstBitIndex = currentLayerIndex++;
			}
			addInputConnection(i, firstBitIndex);
			break;
		case SIGN:
			if (firstSignLayer == NULL){
				firstSignLayer = Factory::newLayer(implementationType, SIGN, hiddenLayersType, functiontype);
				addLayer(firstSignLayer);
				firstSignIndex = currentLayerIndex++;
			}
			addInputConnection(i, firstSignIndex);
			break;
		}
	}
	Layer* unificationLayer;
	if (currentLayerIndex > 1){
		unificationLayer = Factory::newLayer(implementationType, hiddenLayersType, hiddenLayersType, functiontype);

		addLayer(unificationLayer);
	}
	if (firstFloatLayer != NULL){
		firstFloatLayer->setSize(sizeLayers);
		if (currentLayerIndex > 1){
			addLayersConnection(firstFloatIndex, currentLayerIndex);
		}
	}
	if (firstBitLayer != NULL){
		firstBitLayer->setSize(sizeLayers);
		if (currentLayerIndex > 1){
			addLayersConnection(firstBitIndex, currentLayerIndex);
		}
	}
	if (firstSignLayer != NULL){
		firstSignLayer->setSize(sizeLayers);
		if (currentLayerIndex > 1){
			addLayersConnection(firstSignIndex, currentLayerIndex);
		}
	}
	if (currentLayerIndex > 1){
		unificationLayer->setSize(sizeLayers);
		++currentLayerIndex;
	}

	if (sizeLayers < currentLayerIndex){
		cout<<"Warning: there will be "<<currentLayerIndex<<" hidden layers instead of "<<sizeLayers<<"."<<endl;
	}

	unsigned i;
	for (i=currentLayerIndex; i<numLayers; i++){
		Layer* layer = Factory::newLayer(implementationType, hiddenLayersType, hiddenLayersType, functiontype);
		addLayer(layer);
		addLayersConnection(i-1, i);
		layer->setSize(sizeLayers);
	}

	unsigned char offset = 0;
	if (floatOutputSize > 0){
		Layer* layer = Factory::newLayer(implementationType, hiddenLayersType, FLOAT, functiontype);
		addLayer(layer);
		addLayersConnection(i-1, i);
		layer->setSize(floatOutputSize);
		addOutput(i);
		++offset;
	}
	if (bitOutputSize > 0){
		Layer* layer = Factory::newLayer(implementationType, hiddenLayersType, BIT, functiontype);
		addLayer(layer);
		addLayersConnection(i-1, i+offset);
		layer->setSize(bitOutputSize);
		addOutput(i+offset);
	} else if (floatOutputSize == 0){
		//cout<<"The last hidden layer will be the output."<<endl;
		addOutput(i-1);
	}
}

void NeuralNet::createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType)
{
	createFullyConnectedNet(numLayers, sizeLayers, hiddenLayersType, IDENTITY);
}
void NeuralNet::createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype)
{
	if (numberInputs == 0){
		string error = "Cannot create a network with no inputs.";
		throw error;
	}
	for (unsigned i=0; i<numLayers; i++){
		Layer* layer = Factory::newLayer(implementationType, hiddenLayersType, hiddenLayersType, functiontype);
		addLayer(layer);
		layer->setSize(sizeLayers);
	}
	for (unsigned i=0; i<numLayers; i++){
		for(unsigned j=0; j<numberInputs; j++){
			addInputConnection(j, i);
		}
		for(unsigned j=0; j<numLayers; j++){
			if (i != j) {
				addLayersConnection(j, i);
			}
		}
	}
	for (unsigned i=0; i<numLayers; i++){
		layers[i]->resetSize();
	}
	addOutput(numLayers-1);
}

void NeuralNet::save(FILE* stream)
{
	fwrite(&numberInputs, sizeof(unsigned), 1, stream);
	fwrite(&numberLayers, sizeof(unsigned), 1, stream);
	fwrite(&numberOutputs, sizeof(unsigned), 1, stream);

	fwrite(inputsToLayersGraph, sizeof(unsigned char) * numberInputs * numberLayers, 1, stream);
	fwrite(layerConnectionsGraph, sizeof(unsigned char) * numberLayers * numberLayers, 1, stream);
	fwrite(outputLayers, sizeof(int) * numberOutputs, 1, stream);

	for (unsigned i=0; i<numberLayers; i++){
		layers[i]->save(stream);
	}
}

void NeuralNet::load(FILE* stream)
{
	unsigned auxNumberInputs;
	fread(&auxNumberInputs, sizeof(unsigned), 1, stream);

	if (auxNumberInputs != numberInputs){
		char buffer[100];
		sprintf(buffer, "the number of inputs (%d) does not match with the number of inputs of the Neural Net to load (%d).", numberInputs, auxNumberInputs);
		string error = buffer;
		throw error;
	}

	fread(&numberLayers, sizeof(unsigned), 1, stream);
	fread(&numberOutputs, sizeof(unsigned), 1, stream);

	size_t size = sizeof(unsigned char) * numberInputs * numberLayers;
	inputsToLayersGraph = (unsigned char*) mi_malloc(size);
	fread(inputsToLayersGraph, size, 1, stream);

	size = sizeof(unsigned char) * numberLayers * numberLayers;
	layerConnectionsGraph = (unsigned char*) mi_malloc(size);
	fread(layerConnectionsGraph, size, 1, stream);

	size = sizeof(unsigned) * numberOutputs;
	outputLayers = (unsigned*) mi_malloc(size);
	fread(outputLayers, size, 1, stream);

	layers = (Layer**) mi_malloc(sizeof(Layer*) * numberLayers);
	for (unsigned i=0; i<numberLayers; i++){
		layers[i] = Factory::newLayer(this->implementationType);
		layers[i]->load(stream);
	}

	for (unsigned i=0; i<numberInputs; i++){
		for (unsigned j=0; j<numberLayers; j++){
			if (inputsToLayersGraph[(i*numberLayers) + j]) {
				addInputConnection(i, j);
			}
		}
	}

	for (unsigned i=0; i<numberLayers; i++){
		for (unsigned j=0; j<numberLayers; j++){
			if (layerConnectionsGraph[(i*numberLayers) + j]){
				addLayersConnection(i, j);
			}
		}
	}

	outputs = (Vector**) mi_malloc(sizeof(Vector*) * numberOutputs);
	for (unsigned i=0; i<numberOutputs; i++){
		outputs[i] = layers[outputLayers[i]]->getOutput();
	}
}


