#include "neuralNet.h"

NeuralNet::NeuralNet(ImplementationType implementationType) {
	this->implementationType = implementationType;

	layers = NULL;
	layerConnectionsGraph = NULL;
	numberLayers = 0;

	inputs = NULL;
	inputInterfaces = NULL;
	inputsToLayersGraph = NULL;
	numberInputs = 0;

	outputs = NULL;
	outputLayers = NULL;
	numberOutputs = 0;
}

NeuralNet::~NeuralNet() {
	if (layers) {
		for (unsigned i = 0; i < numberLayers; i++) {
			delete (layers[i]);
		}
		mi_free(layers);
	}
	if (layerConnectionsGraph) {
		mi_free(layerConnectionsGraph);
	}
	if (inputs) {
		for (unsigned i = 0; i < numberInputs; i++) {
			delete (inputs[i]);
			//TODO N pensar donde se liberan las primeras interfaces si se usa setInput
			delete (inputInterfaces[i]);
		}
		mi_free(inputs);
		mi_free(inputInterfaces);
	}
	if (inputsToLayersGraph) {
		mi_free(inputsToLayersGraph);
	}
	if (outputs) {
		for (unsigned i = 0; i < numberOutputs; i++) {
			delete (outputs[i]);
		}
		mi_free(outputs);
	}
	if (outputLayers) {
		mi_free(outputLayers);
	}
}

Layer* NeuralNet::getLayer(unsigned pos) {
	return layers[pos];
}

void NeuralNet::calculateOutput() {
	for (unsigned i = 0; i < numberInputs; i++) {
		inputs[i]->copyFromInterface(inputInterfaces[i]);
	}
	for (unsigned i = 0; i < numberLayers; i++) {
		layers[i]->calculateOutput();
	}
	for (unsigned i = 0; i < numberOutputs; i++) {
		layers[outputLayers[i]]->getOutput()->copyToInterface(outputs[i]);
	}
}

void NeuralNet::addLayer(Layer* layer) {
	unsigned newNumberLayers = numberLayers + 1;

	Layer** newLayers = (Layer**) mi_malloc(sizeof(Layer*) * newNumberLayers);
	if (layers) {
		memcpy(newLayers, layers, numberLayers * sizeof(Layer*));
		mi_free(layers);
	}
	layers = newLayers;

	if (numberInputs > 0) {
		if (inputsToLayersGraph) {
			unsigned char* newInputsToLayersGraph = (unsigned char*) mi_malloc(
					sizeof(unsigned char) * numberInputs * newNumberLayers);
			for (unsigned i = 0; i < numberInputs; i++) {
				for (unsigned j = 0; j < numberLayers; j++) {
					newInputsToLayersGraph[(i * numberLayers) + j]
							= inputsToLayersGraph[(i * numberLayers) + j];
				}
				newInputsToLayersGraph[(i * numberLayers) + numberLayers] = 0;
			}
			mi_free(inputsToLayersGraph);
			inputsToLayersGraph = newInputsToLayersGraph;
		} else {
			inputsToLayersGraph = (unsigned char*) mi_malloc(
					sizeof(unsigned char) * numberInputs * newNumberLayers);
			for (unsigned i = 0; i < numberInputs * newNumberLayers; i++) {
				inputsToLayersGraph[i] = 0;
			}
		}
	}

	unsigned char* newLayerConnectionsGraph = (unsigned char*) mi_malloc(
			sizeof(unsigned char) * newNumberLayers * newNumberLayers);
	for (unsigned i = 0; i < newNumberLayers; i++) {
		for (unsigned j = 0; j < newNumberLayers; j++) {
			if (i == numberLayers || j == numberLayers) {
				newLayerConnectionsGraph[(i * newNumberLayers) + j] = 0;
			} else {
				newLayerConnectionsGraph[(i * newNumberLayers) + j]
						= layerConnectionsGraph[(i * numberLayers) + j];
			}
		}
	}
	if (layerConnectionsGraph) {
		mi_free(layerConnectionsGraph);
	}
	layerConnectionsGraph = newLayerConnectionsGraph;

	layers[numberLayers++] = layer;
}

void NeuralNet::addLayer(unsigned size, VectorType destinationType, FunctionType functiontype) {
	Layer* layer = new Layer(size, destinationType, functiontype, implementationType);
	addLayer(layer);
}

Interface* NeuralNet::createInput(unsigned size, VectorType vectorType) {
	Vector* input = Factory::newVector(size, vectorType, implementationType);
	Interface* interface = new Interface(size, vectorType);

	unsigned newNumberInputs = numberInputs + 1;

	Vector** newInputs =
			(Vector**) mi_malloc(sizeof(Vector*) * newNumberInputs);
	Interface** newInterfaces = (Interface**) mi_malloc(sizeof(Interface*)
			* newNumberInputs);
	if (inputs) {
		memcpy(newInputs, inputs, numberInputs * sizeof(Vector*));
		memcpy(newInterfaces, inputInterfaces, numberInputs
				* sizeof(Interface*));
		mi_free(inputs);
		mi_free(inputInterfaces);
	}
	inputs = newInputs;
	inputs[numberInputs] = input;
	inputInterfaces = newInterfaces;
	inputInterfaces[numberInputs] = interface;

	if (numberLayers > 0) {
		if (inputsToLayersGraph) {
			unsigned char* newInputsToLayersGraph = (unsigned char*) mi_malloc(
					sizeof(unsigned char) * newNumberInputs * numberLayers);
			for (unsigned i = 0; i < numberInputs; i++) {
				for (unsigned j = 0; j < numberLayers; j++) {
					newInputsToLayersGraph[(i * numberLayers) + j]
							= inputsToLayersGraph[(i * numberLayers) + j];
				}
			}
			for (unsigned j = 0; j < numberLayers; j++) {
				newInputsToLayersGraph[(numberInputs * numberLayers) + j] = 0;
			}
			mi_free(inputsToLayersGraph);
			inputsToLayersGraph = newInputsToLayersGraph;
		} else {
			inputsToLayersGraph = (unsigned char*) mi_malloc(
					sizeof(unsigned char) * newNumberInputs * numberLayers);
			for (unsigned i = 0; i < newNumberInputs * numberLayers; i++) {
				inputsToLayersGraph[i] = 0;
			}
		}
	}
	++numberInputs;
	return interface;
}

Interface* NeuralNet::getInput(unsigned pos) {
	return inputInterfaces[pos];
}

void NeuralNet::setInput(unsigned pos, Interface* input) {
	inputInterfaces[pos] = input;
}

unsigned NeuralNet::getNumInputs() {
	return numberInputs;
}

Interface* NeuralNet::createOutput(unsigned layerPos) {
	if (layerPos >= numberLayers) {
		char buffer[100];
		sprintf(
				buffer,
				"Cannot set the Layer in position %d as an output: there are just %d Layers.",
				layerPos, numberLayers);
		std::string error = buffer;
		throw error;
	}

	Interface** newOutputs = (Interface**) mi_malloc(sizeof(Interface*)
			* (numberOutputs + 1));
	if (outputs) {
		memcpy(newOutputs, outputs, numberOutputs * sizeof(Interface*));
		mi_free(outputs);
	}
	outputs = newOutputs;

	unsigned* newOutputLayers = (unsigned*) mi_malloc(sizeof(unsigned)
			* (numberOutputs + 1));
	if (outputLayers) {
		memcpy(newOutputLayers, outputLayers, numberOutputs * sizeof(unsigned));
		mi_free(outputLayers);
	}
	outputLayers = newOutputLayers;

	outputLayers[numberOutputs] = layerPos;
	outputs[numberOutputs] = new Interface(
			layers[layerPos]->getOutput()->getSize(),
			layers[layerPos]->getOutput()->getVectorType());

	return outputs[numberOutputs++];
}

Interface* NeuralNet::getOutput(unsigned outputPos) {
	if (outputPos >= numberLayers) {
		char buffer[100];
		sprintf(
				buffer,
				"Cannot access the output in position %d: there are just %d outputs.",
				outputPos, numberLayers);
		std::string error = buffer;
		throw error;
	}
	return outputs[outputPos];
}

unsigned NeuralNet::getNumOutputs() {
	return numberOutputs;
}

void NeuralNet::randomWeighs(float range) {
	for (unsigned i = 0; i < numberLayers; i++) {
		layers[i]->randomWeighs(range);
	}
}

void NeuralNet::addInputConnection(unsigned sourceInputPos,
		unsigned destinationLayerPos) {
	if (!inputsToLayersGraph) {
		std::string error = "The inputs To Layers Graph is not set yet.";
		throw error;
	}
	if (sourceInputPos >= numberInputs) {
		char buffer[100];
		sprintf(
				buffer,
				"Cannot connect input in position %d: there are just %d inputs.",
				sourceInputPos, numberInputs);
		std::string error = buffer;
		throw error;
	}
	if (destinationLayerPos >= numberLayers) {
		char buffer[100];
		sprintf(
				buffer,
				"Cannot connect an input with the Layer in position %d: there are just %d Layers.",
				destinationLayerPos, numberLayers);
		std::string error = buffer;
		throw error;
	}

	layers[destinationLayerPos]->addInput(inputs[sourceInputPos]);
	inputsToLayersGraph[(sourceInputPos * numberLayers) + destinationLayerPos] = 1;
}

void NeuralNet::addLayersConnection(unsigned sourceLayerPos,
		unsigned destinationLayerPos) {
	if (sourceLayerPos >= numberLayers || destinationLayerPos >= numberLayers) {
		char buffer[100];
		sprintf(
				buffer,
				"Cannot connect Layer in position %d with Layer in position %d: there are just %d Layers.",
				sourceLayerPos, destinationLayerPos, numberLayers);
		std::string error = buffer;
		throw error;
	}
	if (sourceLayerPos == destinationLayerPos) {
		char buffer[100];
		sprintf(buffer, "Cannot connect Layer in position %d with itself.",
				sourceLayerPos);
		std::string error = buffer;
		throw error;
	}

	layers[destinationLayerPos]->addInput(layers[sourceLayerPos]->getOutput());
	layerConnectionsGraph[(sourceLayerPos * numberLayers) + destinationLayerPos]
			= 1;
}

void NeuralNet::createFeedForwardNet(unsigned numLayers, unsigned sizeLayers,
		VectorType hiddenLayersType, FunctionType functiontype) {
	if (numberInputs == 0) {
		std::string error = "Cannot create a network with no inputs.";
		throw error;
	}

	addLayer(sizeLayers, hiddenLayersType, functiontype);
	for (unsigned i = 0; i < numberInputs; i++) {
		addInputConnection(i, 0);
	}

	for (unsigned i = 1; i < numLayers; i++) {
		addLayer(sizeLayers, hiddenLayersType, functiontype);
		addLayersConnection(i - 1, i);
	}

	createOutput(this->numberLayers - 1);
}

void NeuralNet::createFullyConnectedNet(unsigned numLayers,
		unsigned sizeLayers, VectorType hiddenLayersType,
		FunctionType functiontype) {
	if (numberInputs == 0) {
		std::string error = "Cannot create a network with no inputs.";
		throw error;
	}
	for (unsigned i = 0; i < numLayers; i++) {
		addLayer(sizeLayers, hiddenLayersType, functiontype);
	}
	for (unsigned i = 0; i < numLayers; i++) {
		for (unsigned j = 0; j < numberInputs; j++) {
			addInputConnection(j, i);
		}
		for (unsigned j = 0; j < numLayers; j++) {
			if (i != j) {
				addLayersConnection(j, i);
			}
		}
	}
	createOutput(numLayers - 1);
}

//TODO N rehacer save y load
void NeuralNet::save(FILE* stream) {
	fwrite(&numberInputs, sizeof(unsigned), 1, stream);
	fwrite(&numberLayers, sizeof(unsigned), 1, stream);
	fwrite(&numberOutputs, sizeof(unsigned), 1, stream);

	for (unsigned i = 0; i < numberLayers; i++) {
		layers[i]->save(stream);
	}

	fwrite(inputsToLayersGraph, sizeof(unsigned char) * numberInputs
			* numberLayers, 1, stream);
	fwrite(layerConnectionsGraph, sizeof(unsigned char) * numberLayers
			* numberLayers, 1, stream);
	fwrite(outputLayers, sizeof(int) * numberOutputs, 1, stream);

	for (unsigned i = 0; i < numberLayers; i++) {
		layers[i]->saveWeighs(stream);
	}
}

void NeuralNet::load(FILE* stream) {
	unsigned auxNumberInputs;
	fread(&auxNumberInputs, sizeof(unsigned), 1, stream);

	if (auxNumberInputs != numberInputs) {
		char buffer[100];
		sprintf(
				buffer,
				"the number of inputs (%d) does not match with the number of inputs of the Neural Net to load (%d).",
				numberInputs, auxNumberInputs);
		std::string error = buffer;
		throw error;
	}

	fread(&numberLayers, sizeof(unsigned), 1, stream);
	fread(&numberOutputs, sizeof(unsigned), 1, stream);

	layers = (Layer**) ((mi_malloc(sizeof(Layer*) * numberLayers)));
	for (unsigned i = 0; i < numberLayers; i++) {
		layers[i] = new Layer(stream, implementationType);
	}

	loadGraphs(stream);
	stablishConnections();

	for (unsigned i = 0; i < numberLayers; i++) {
		layers[i]->loadWeighs(stream);
	}
}

void NeuralNet::loadGraphs(FILE* stream) {
	size_t size = sizeof(unsigned char) * numberInputs * numberLayers;
	inputsToLayersGraph = (unsigned char*) ((mi_malloc(size)));
	fread(inputsToLayersGraph, size, 1, stream);

	size = sizeof(unsigned char) * numberLayers * numberLayers;
	layerConnectionsGraph = (unsigned char*) ((mi_malloc(size)));
	fread(layerConnectionsGraph, size, 1, stream);

	size = sizeof(unsigned) * numberOutputs;
	outputLayers = (unsigned *) ((mi_malloc(size)));
	fread(outputLayers, size, 1, stream);
}

void NeuralNet::stablishConnections() {
	for (unsigned i = 0; i < numberInputs; i++) {
		for (unsigned j = 0; j < numberLayers; j++) {
			if (inputsToLayersGraph[(i * numberLayers) + j]) {
				addInputConnection(i, j);
			}
		}
	}
	for (unsigned i = 0; i < numberLayers; i++) {
		for (unsigned j = 0; j < numberLayers; j++) {
			if (layerConnectionsGraph[(i * numberLayers) + j]) {
				addLayersConnection(i, j);
			}
		}
	}
	outputs = (Interface**) (mi_malloc(sizeof(Interface*) * numberOutputs));
	for (unsigned i = 0; i < numberOutputs; i++) {
		Vector *outputVector = layers[outputLayers[i]]->getOutput();
		outputs[i] = new Interface(outputVector->getSize(),
				outputVector->getVectorType());
	}
}
