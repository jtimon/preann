#include "neuralNet.h"

NeuralNet::NeuralNet()
{
	layers = NULL;
	layerConnectionsGraph = NULL;
	numberLayers = 0;
	maxLayers = 0;

	inputs = NULL;
	inputsToLayersGraph = NULL;
	numberInputs = 0;
	this->maxInputs = 0;

	outputs = NULL;
	outputLayers = NULL;
	numberOutputs = 0;
	maxOutputs = 0;
}

NeuralNet::NeuralNet(unsigned maxInputs, unsigned maxLayers, unsigned maxOutputs)
{
	layers = (Layer**) malloc(sizeof(Layer*) * maxLayers);
	layerConnectionsGraph = (unsigned char*) malloc(sizeof(unsigned char) * maxLayers * maxLayers);
	numberLayers = 0;
	this->maxLayers = maxLayers;

	inputs = (Vector**) malloc(sizeof(Vector*) * maxInputs);
	inputsToLayersGraph = (unsigned char*) malloc(sizeof(unsigned char) * maxInputs * maxLayers);
	numberInputs = 0;
	this->maxInputs = maxInputs;

	outputs = (Vector**) malloc(sizeof(Vector*) * maxOutputs);
	outputLayers = (int*) malloc(sizeof(int) * maxOutputs);
	numberOutputs = 0;
	this->maxOutputs = maxOutputs;

	for (unsigned i=0; i < maxLayers; i++){
		layers[i] = NULL;
		for (unsigned j=0; j < maxLayers; j++){
			layerConnectionsGraph[getPosInGraph(i, j)] = 0;
		}
	}
	for (unsigned i=0; i < maxInputs; i++){
		layers[i] = NULL;
		for (unsigned j=0; j < maxLayers; j++){
			inputsToLayersGraph[getPosInGraph(i, j)] = 0;
		}
	}
	for (unsigned i=0; i < maxLayers; i++){
		outputs[i] = NULL;
		outputLayers[i] = -1;
	}
}

NeuralNet::~NeuralNet()
{/*
	if (layers != NULL) {
		for (unsigned i=0; i < numberLayers; i++){
			layers[i]->freeLayer();
			//TODO descomentar y evitar que pete
			//delete (layers[i]);
		}
		delete[] layers;
	}
	if (layerConnectionsGraph != NULL)
		delete[] layerConnectionsGraph;

	if (inputs != NULL)
		delete[] inputs;
	//TODO descomentar y evitar que pete
//	if (inputsToLayersGraph != NULL)
//		delete[] inputsToLayersGraph;

	if (outputs != NULL)
		delete[] outputs;
	if (outputLayers != NULL)
		delete[] outputLayers;*/

}

void NeuralNet::freeNeuralNet()
{
	if (layers) {
		for (unsigned i=0; i < numberLayers; i++) {
			layers[i]->freeLayer();
			//TODO descomentar y evitar que pete
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

Layer* NeuralNet::newLayer(){
	return new Layer();
}

Layer* NeuralNet::newLayer(VectorType inputType, VectorType outputType, FunctionType functionType){
	return new Layer(inputType, outputType, functionType);
}

Vector* NeuralNet::newVector(unsigned size, VectorType vectorType)
{
	return new Vector(size, vectorType);
}

void NeuralNet::calculateOutput()
{
	for (unsigned i=0; i < numberLayers; i++){
		cout<<"calculando capa "<<i<<endl;
		layers[i]->randomWeighs(20);
		Vector* aa = layers[i]->newVector(10, BIT); //TODO quitar
		layers[i]->calculateOutput();
		delete(aa);
	}
}

void NeuralNet::changeMaxInputs(unsigned newMaxInputs)
{
	Vector** newInputs = (Vector**) malloc(sizeof(Vector*) * newMaxInputs);
	unsigned char* newInputsToLayersGraph = new unsigned char[newMaxInputs * maxLayers];

	unsigned minInputs;
	if (newMaxInputs > numberInputs){
		minInputs = numberInputs;
		for (unsigned i=minInputs; i < newMaxInputs; i++){
			newInputs[i] = NULL;
		}
	} else {
		minInputs = newMaxInputs;
	}

	for (unsigned i=0; i < minInputs; i++){
		for (unsigned j=0; j < numberLayers; j++){
			newInputsToLayersGraph[getPosInGraph(i, j)] = inputsToLayersGraph[getPosInGraph(i, j)];
		}
	}
	for (unsigned i=minInputs; i < newMaxInputs; i++){
		for (unsigned j=0; j < maxLayers; j++){
			newInputsToLayersGraph[getPosInGraph(i, j)] = 0;
		}
	}

	if (minInputs > 0) {
		memcpy(newInputs, inputs, minInputs * sizeof(Vector*));
		free(inputs);
		free(inputsToLayersGraph);
	}
	inputs = newInputs;
	inputsToLayersGraph = newInputsToLayersGraph;
	maxInputs = newMaxInputs;
}

void NeuralNet::changeMaxLayers(unsigned newMaxLayers)
{
	Layer** newLayers = (Layer**) malloc(sizeof(Layer*) * newMaxLayers);
	unsigned char* newLayerConnectionsGraph = (unsigned char*) malloc(sizeof(unsigned char) * newMaxLayers*newMaxLayers);
	unsigned char* newInputsToLayersGraph = new unsigned char[maxInputs * newMaxLayers];

	unsigned minLayers;
	if (newMaxLayers > numberLayers){
		minLayers = numberLayers;
		for (unsigned i=minLayers; i < newMaxLayers; i++){
			newLayers[i] = NULL;
		}
	} else {
		minLayers = newMaxLayers;
		for (unsigned i=minLayers; i < numberLayers; i++){
			delete (layers[i]);
		}
	}
	for (unsigned i=0; i < newMaxLayers; i++){
		for (unsigned j=0; j < newMaxLayers; j++){
			if (i >= minLayers || j >= minLayers){
				newLayerConnectionsGraph[(i * newMaxLayers) + j] = 0;
			} else {
				newLayerConnectionsGraph[(i * newMaxLayers) + j] = layerConnectionsGraph[getPosInGraph(i, j)];
			}
		}
	}
	for (unsigned i=0; i < maxInputs; i++){
		for (unsigned j=0; j < minLayers; j++){
			newInputsToLayersGraph[(i * newMaxLayers) + j] = inputsToLayersGraph[getPosInGraph(i, j)];
		}
	}
	for (unsigned i=0; i < maxInputs; i++){
		for (unsigned j=minLayers; j < newMaxLayers; j++){
			newInputsToLayersGraph[(i * newMaxLayers) + j] = 0;
		}
	}
	if (minLayers > 0) {
		memcpy(newLayers, layers, minLayers * sizeof(Layer*));
		free(layers);
		free(layerConnectionsGraph);
		free(inputsToLayersGraph);
	}
	layers = newLayers;
	layerConnectionsGraph = newLayerConnectionsGraph;
	inputsToLayersGraph = newInputsToLayersGraph;
	maxLayers = newMaxLayers;
}

void NeuralNet::changeMaxOuputs(unsigned newMaxOutputs)
{
	Vector** newOutputs = (Vector**) malloc(sizeof(Vector*) * newMaxOutputs);
	int* newOutputLayers = (int*) malloc(sizeof(int) * newMaxOutputs);

	unsigned minOutputs;
	if (newMaxOutputs > numberOutputs){
		minOutputs = numberOutputs;
	} else {
		minOutputs = newMaxOutputs;
	}
	if (minOutputs > 0){
		memcpy(newOutputs, outputs, minOutputs * sizeof(Vector*));
		memcpy(newOutputLayers, outputLayers, minOutputs * sizeof(int));
		free(outputs);
		free(outputLayers);
	}
	for(unsigned i=minOutputs; i < newMaxOutputs; i++) {
		newOutputs[i] = NULL;
		newOutputLayers[i] = -1;
	}

	outputs = newOutputs;
	outputLayers = newOutputLayers;
	maxOutputs = newMaxOutputs;
}

unsigned NeuralNet::getPosInGraph(unsigned source, unsigned destination)
{
	return (source*maxLayers) + destination;
}

void NeuralNet::addInput(Vector* input)
{
	if (numberInputs == maxInputs){
		changeMaxInputs(numberInputs+1);
	}
	inputs[numberInputs] = input;
	numberInputs++;
}

Vector* NeuralNet::getOutput(unsigned outputPos)
{
	if (outputPos >= numberOutputs){
		cout<<"Error: cannot access to output in position "<<outputPos<<". The number of outputs is "<<numberOutputs<<"."<<endl;
		return NULL;
	}
	return outputs[outputPos];
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

void NeuralNet::addLayer(Layer* layer)
{
	if (numberLayers == maxLayers){
		changeMaxLayers(maxLayers+1);
	}
	layers[numberLayers++] = layer;
}

void NeuralNet::addLayer(unsigned  size, VectorType sourceType, VectorType destinationType, FunctionType functiontype)
{
	Layer* layer = newLayer(sourceType, destinationType, functiontype);
	addLayer(layer);
	layer->setSize(size);
}

void NeuralNet::addLayer(unsigned  size, VectorType sourceType, VectorType destinationType)
{
	addLayer(size, sourceType, destinationType, IDENTITY);
}

void NeuralNet::addInputConnection(unsigned  sourceInputPos, unsigned  destinationLayerPos)
{
	if (sourceInputPos > numberInputs){
		cout<<"Error: cannot connect input "<<sourceInputPos<<": there are just "<<numberInputs<<" inputs"<<endl;
	}
	else if (destinationLayerPos > numberLayers){
		cout<<"Error: cannot connect layer "<<destinationLayerPos<<": there are just "<<numberLayers<<" layers"<<endl;
	} else {
		inputsToLayersGraph[getPosInGraph(sourceInputPos, destinationLayerPos)] =
					layers[destinationLayerPos]->addInput(inputs[sourceInputPos]);
	}
}

void NeuralNet::addLayersConnection(unsigned  sourceLayerPos, unsigned  destinationLayerPos)
{
	if (sourceLayerPos > numberLayers){
		cout<<"Error: cannot connect layer "<<sourceLayerPos<<": there are just "<<numberLayers<<" layers"<<endl;
	}
	else if (destinationLayerPos > numberLayers){
		cout<<"Error: cannot connect layer "<<destinationLayerPos<<": there are just "<<numberLayers<<" layers"<<endl;
	} else if (sourceLayerPos == destinationLayerPos){
		cout<<"Error: cannot connect layer "<<destinationLayerPos<<" with itself."<<endl;
	} else {
		layerConnectionsGraph[getPosInGraph(sourceLayerPos, destinationLayerPos)] =
				layers[destinationLayerPos]->addInput(layers[sourceLayerPos]->getOutput());
	}
}

void NeuralNet::setLayerAsOutput(unsigned layerPos)
{
	if (layerPos > numberLayers){
		cout<<"Error: cannot access to layer "<<layerPos<<": there are just "<<numberLayers<<" layers"<<endl;
	} else {

		if (numberOutputs == maxOutputs){
			changeMaxOuputs(numberOutputs+1);
		}
		outputLayers[numberOutputs] = layerPos;
		outputs[numberOutputs++] = layers[layerPos]->getOutput();
	}
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
		cout<<"Error: you have to specify the inputs before creating the network."<<endl;
	} else {
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
					firstFloatLayer = newLayer(FLOAT, hiddenLayersType, functiontype);
					addLayer(firstFloatLayer);
					firstFloatIndex = currentLayerIndex++;
				}
				addInputConnection(i, firstFloatIndex);
				break;
			case BIT:
				if (firstBitLayer == NULL){
					firstBitLayer = newLayer(BIT, hiddenLayersType, functiontype);
					addLayer(firstBitLayer);
					firstBitIndex = currentLayerIndex++;
				}
				addInputConnection(i, firstBitIndex);
				break;
			case SIGN:
				if (firstSignLayer == NULL){
					firstSignLayer = newLayer(SIGN, hiddenLayersType, functiontype);
					addLayer(firstSignLayer);
					firstSignIndex = currentLayerIndex++;
				}
				addInputConnection(i, firstSignIndex);
				break;
			}
		}
		Layer* unificationLayer;
		if (currentLayerIndex > 1){
			unificationLayer = newLayer(hiddenLayersType, hiddenLayersType, functiontype);
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
			Layer* layer = newLayer(hiddenLayersType, hiddenLayersType, functiontype);
			addLayer(layer);
			addLayersConnection(i-1, i);
			layer->setSize(sizeLayers);
		}

		unsigned char offset = 0;
		if (floatOutputSize > 0){

			Layer* layer = newLayer(hiddenLayersType, FLOAT, functiontype);
			addLayer(layer);
			addLayersConnection(i-1, i);
			layer->setSize(floatOutputSize);
			setLayerAsOutput(i);
			++offset;
		}
		if (bitOutputSize > 0){
			Layer* layer = newLayer(hiddenLayersType, BIT, functiontype);
			addLayer(layer);
			addLayersConnection(i-1, i+offset);
			layer->setSize(bitOutputSize);
			setLayerAsOutput(i+offset);
		} else if (floatOutputSize == 0){
			//cout<<"The last hidden layer will be the output."<<endl;
			setLayerAsOutput(i-1);
		}
	}
}

void NeuralNet::createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType)
{
	createFullyConnectedNet(numLayers, sizeLayers, hiddenLayersType, IDENTITY);
}
void NeuralNet::createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype)
{
	if (numberInputs == 0){
		cout<<"Error: you have to specify the inputs before creating the network."<<endl;
	} else {
		for (unsigned i=0; i<numLayers; i++){
			Layer* layer = newLayer(hiddenLayersType, hiddenLayersType, functiontype);
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
		setLayerAsOutput(numLayers-1);
	}
}

void NeuralNet::save(FILE* stream)
{
	fwrite(&numberInputs, sizeof(unsigned), 1, stream);
	fwrite(&numberLayers, sizeof(unsigned), 1, stream);
	fwrite(&numberOutputs, sizeof(unsigned), 1, stream);

	unsigned char* auxInputsToLayersGraph = new unsigned char[numberInputs * numberLayers];
	for (unsigned i=0; i<numberInputs; i++){
		for (unsigned j=0; j<numberLayers; j++){
			auxInputsToLayersGraph[(i*numberLayers) + j] = inputsToLayersGraph[(i*maxLayers) + j];
		}
	}
	fwrite(auxInputsToLayersGraph, sizeof(unsigned char) * numberInputs * numberLayers, 1, stream);

	unsigned char* auxLayerConnectionsGraph = new unsigned char[numberLayers * numberLayers];
	for (unsigned i=0; i<numberLayers; i++){
		for (unsigned j=0; j<numberLayers; j++){
			auxLayerConnectionsGraph[(i*numberLayers) + j] = layerConnectionsGraph[(i*maxLayers) + j];
		}
	}
	fwrite(auxLayerConnectionsGraph, sizeof(unsigned char) * numberLayers * numberLayers, 1, stream);

	int* auxOutputLayers = new int[numberOutputs];
	for (unsigned i=0; i<numberOutputs; i++){
		auxOutputLayers[i] = outputLayers[i];
	}
	fwrite(auxOutputLayers, sizeof(int) * numberOutputs, 1, stream);

	for (unsigned i=0; i<numberLayers; i++){
		layers[i]->save(stream);
	}
}

void NeuralNet::load(FILE* stream)
{
	unsigned auxNumberInputs;
	fread(&auxNumberInputs, sizeof(unsigned), 1, stream);
	if (auxNumberInputs != numberInputs){
		cout<<"Error: the number of inputs does not match with the Neural Net to load."<<endl;
	} else {

		fread(&numberLayers, sizeof(unsigned), 1, stream);
		fread(&numberOutputs, sizeof(unsigned), 1, stream);
		maxInputs = numberInputs;
		maxLayers = numberLayers;
		maxOutputs = numberOutputs;

		inputsToLayersGraph = (unsigned char*) malloc(sizeof(unsigned char) * numberInputs * numberLayers);
		fread(inputsToLayersGraph, sizeof(unsigned char) * numberInputs * numberLayers, 1, stream);

		layerConnectionsGraph = (unsigned char*) malloc(sizeof(unsigned char) * numberLayers * numberLayers);
		fread(layerConnectionsGraph, sizeof(unsigned char) * numberLayers * numberLayers, 1, stream);

		outputLayers = (int*) malloc(sizeof(int) * numberOutputs);
		fread(outputLayers, sizeof(int) * numberOutputs, 1, stream);

		layers = new Layer*[numberLayers];
		for (unsigned i=0; i<numberLayers; i++){
			layers[i] = newLayer();
			layers[i]->load(stream);
		}

		for (unsigned i=0; i<numberInputs; i++){
			for (unsigned j=0; j<numberLayers; j++){
				if (inputsToLayersGraph[(i*maxLayers) + j]) {
					addInputConnection(i, j);
				}
			}
		}

		for (unsigned i=0; i<numberLayers; i++){
			for (unsigned j=0; j<numberLayers; j++){
				if (layerConnectionsGraph[(i*maxLayers) + j]){
					addLayersConnection(i, j);
				}
			}
		}

		outputs = (Vector**) malloc(sizeof(Vector*) * numberOutputs);
		for (unsigned i=0; i<numberOutputs; i++){
			outputs[i] = layers[outputLayers[i]]->getOutput();
		}
	}
}


