#include "neuralNet.h"

NeuralNet::NeuralNet(ImplementationType implementationType)
{
	this->implementationType = implementationType;

	layers = NULL;
	layerConnectionsGraph = NULL;
	numberLayers = 0;

	inputLayers = NULL;
	numberInputs = 0;
}

NeuralNet::~NeuralNet()
{
	if (layers)
	{
		for (unsigned i = 0; i < numberLayers; i++)
		{
			delete (layers[i]);
		}
		mi_free(layers);
	}
	if (layerConnectionsGraph)
	{
		mi_free(layerConnectionsGraph);
	}
	if (inputLayers)
	{
		mi_free(inputLayers);
	}
}

Layer* NeuralNet::getLayer(unsigned pos)
{
	return layers[pos];
}

void NeuralNet::calculateOutput()
{
	for (unsigned i = 0; i < numberLayers; i++)
	{
		layers[i]->calculateOutput();
	}
}

void NeuralNet::addLayer(Layer* layer)
{
	unsigned newNumberLayers = numberLayers + 1;

	Layer** newLayers = (Layer**)mi_malloc(sizeof(Layer*) * newNumberLayers);
	if (layers)
	{
		memcpy(newLayers, layers, numberLayers * sizeof(Layer*));
		mi_free(layers);
	}
	layers = newLayers;

	unsigned char* newLayerConnectionsGraph = (unsigned char*)mi_malloc(
			sizeof(unsigned char) * newNumberLayers * newNumberLayers);
	for (unsigned i = 0; i < newNumberLayers; i++)
	{
		for (unsigned j = 0; j < newNumberLayers; j++)
		{
			if (i == numberLayers || j == numberLayers)
			{
				newLayerConnectionsGraph[(i * newNumberLayers) + j] = 0;
			}
			else
			{
				newLayerConnectionsGraph[(i * newNumberLayers) + j]
						= layerConnectionsGraph[(i * numberLayers) + j];
			}
		}
	}
	if (layerConnectionsGraph)
	{
		mi_free(layerConnectionsGraph);
	}
	layerConnectionsGraph = newLayerConnectionsGraph;

	layers[numberLayers++] = layer;
}

void NeuralNet::addLayer(unsigned size, VectorType destinationType,
		FunctionType functiontype)
{
	addLayer(new Layer(size, destinationType, functiontype, getImplementationType()));
}

void NeuralNet::addInputLayer(unsigned size, VectorType vectorType)
{

	unsigned* newInputLayers = (unsigned*)mi_malloc(sizeof(unsigned)
			* (numberInputs + 1));
	if (inputLayers)
	{
		memcpy(newInputLayers, inputLayers, numberInputs * sizeof(unsigned));
		mi_free(inputLayers);
	}
	inputLayers = newInputLayers;
	inputLayers[numberInputs++] = numberLayers;

	addLayer(new InputLayer(size, vectorType, getImplementationType()));
}

void NeuralNet::updateInput(unsigned inputPos, Interface* input)
{
	if (inputPos > numberInputs)
	{
		char buffer[100];
		sprintf(
				buffer,
				"Cannot get the Input in position %d: there are just %d Inputs.",
				inputPos, numberInputs);
		std::string error = buffer;
		throw error;
	}
	return ((InputLayer*)(layers[inputLayers[inputPos]]))->getInputInterface()->copyFromFast(input);
}

unsigned NeuralNet::getNumInputs()
{
	return numberInputs;
}

unsigned char NeuralNet::isInputLayer(unsigned layerPos)
{
	if (layerPos >= numberLayers)
	{
		char buffer[100];
		sprintf(
				buffer,
				"Cannot access the Layer in position %d: there are just %d layers.",
				layerPos, numberLayers);
		std::string error = buffer;
		throw error;
	}
	for (unsigned i = 0; i < numberInputs; ++i)
	{
		if (inputLayers[i] == layerPos)
		{
			return 1;
		}
	}
	return 0;
}

Interface* NeuralNet::getOutput(unsigned layerPos)
{
	if (layerPos >= numberLayers)
	{
		char buffer[100];
		sprintf(
				buffer,
				"Cannot access the output in position %d: there are just %d layers.",
				layerPos, numberLayers);
		std::string error = buffer;
		throw error;
	}
	return layers[ layerPos ]->getOutputInterface();
}

unsigned NeuralNet::getNumLayers()
{
	return numberLayers;
}

void NeuralNet::randomWeighs(float range)
{
	for (unsigned i = 0; i < numberLayers; i++)
	{
		layers[i]->randomWeighs(range);
	}
}

void NeuralNet::addLayersConnection(unsigned sourceLayerPos,
		unsigned destinationLayerPos)
{
	if (sourceLayerPos >= numberLayers || destinationLayerPos >= numberLayers)
	{
		char buffer[100];
		sprintf(
				buffer,
				"Cannot connect Layer in position %d with Layer in position %d: there are just %d Layers.",
				sourceLayerPos, destinationLayerPos, numberLayers);
		std::string error = buffer;
		throw error;
	}

	layers[destinationLayerPos]->addInput(layers[sourceLayerPos]->getOutput());
	layerConnectionsGraph[(sourceLayerPos * numberLayers) + destinationLayerPos] = 1;
}

void NeuralNet::createFeedForwardNet(unsigned inputSize, VectorType inputType,
		unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType,
		FunctionType functiontype)
{
	addInputLayer(inputSize, inputType);
	for (unsigned i = 0; i < numLayers; i++)
	{
		addLayer(sizeLayers, hiddenLayersType, functiontype);
		addLayersConnection(i, i + 1);
	}
}

ImplementationType NeuralNet::getImplementationType()
{
	if(numberLayers != 0){
		return layers[0]->getImplementationType();
	} else {

	}
	return implementationType;
}

void NeuralNet::createFullyConnectedNet(unsigned inputSize,
		VectorType inputType, unsigned numLayers, unsigned sizeLayers,
		VectorType hiddenLayersType, FunctionType functiontype)
{
	addInputLayer(inputSize, inputType);
	for (unsigned i = 0; i < numLayers; i++)
	{
		addLayer(sizeLayers, hiddenLayersType, functiontype);
	}

	for (unsigned src = 0; src <= numLayers; src++)
	{
		for (unsigned dest = 1; dest <= numLayers; dest++)
		{
			addLayersConnection(src, dest);
		}
	}
}

void NeuralNet::save(FILE* stream)
{
	fwrite(&numberInputs, sizeof(unsigned), 1, stream);
	fwrite(&numberLayers, sizeof(unsigned), 1, stream);

	for (unsigned i = 0; i < numberLayers; i++)
	{
		layers[i]->save(stream);
	}

	fwrite(inputLayers, sizeof(unsigned) * numberInputs, 1, stream);
	fwrite(layerConnectionsGraph, sizeof(unsigned char) * numberLayers
			* numberLayers, 1, stream);

	for (unsigned i = 0; i < numberLayers; i++)
	{
		layers[i]->saveWeighs(stream);
	}
}

void NeuralNet::load(FILE* stream)
{
	fread(&numberInputs, sizeof(unsigned), 1, stream);
	fread(&numberLayers, sizeof(unsigned), 1, stream);

	layers = (Layer**)((mi_malloc(sizeof(Layer*) * numberLayers)));
	for (unsigned i = 0; i < numberLayers; i++)
	{
		layers[i] = new Layer(stream, getImplementationType());
	}

	loadGraphs(stream);
	stablishConnections();

	for (unsigned i = 0; i < numberLayers; i++)
	{
		layers[i]->loadWeighs(stream);
	}
}

void NeuralNet::loadGraphs(FILE* stream)
{
	size_t size = sizeof(unsigned) * numberInputs;
	inputLayers = (unsigned*)(mi_malloc(size));
	fread(inputLayers, size, 1, stream);

	size = sizeof(unsigned char) * numberLayers * numberLayers;
	layerConnectionsGraph = (unsigned char*)(mi_malloc(size));
	fread(layerConnectionsGraph, size, 1, stream);
}

void NeuralNet::stablishConnections()
{
	for (unsigned i = 0; i < numberInputs; i++)
	{
		unsigned layerPos = inputLayers[i];
		Vector* output = layers[layerPos]->getOutput();
		Layer* inputLayer = new InputLayer(output->getSize(),
				output->getVectorType(), getImplementationType());
		inputLayer->getOutput()->copyFrom(output);
		delete (layers[layerPos]);
		layers[layerPos] = inputLayer;
	}

	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < numberLayers; j++)
		{
			if (layerConnectionsGraph[(i * numberLayers) + j])
			{
				addLayersConnection(i, j);
			}
		}
	}
}
