#include "neuralNet.h"

NeuralNet::NeuralNet(ImplementationType implementationType)
{
	this->implementationType = implementationType;

	connectionsGraph = NULL;
}

NeuralNet::~NeuralNet()
{
	for (unsigned i = 0; i < layers.size(); i++)
	{
		delete (layers[i]);
	}
	layers.clear();

	if (connectionsGraph)
	{
		MemoryManagement::free(connectionsGraph);
	}
	inputs.clear();
}

Layer* NeuralNet::getLayer(unsigned pos)
{
	return layers[pos];
}

void NeuralNet::calculateOutput()
{
	for (unsigned i = 0; i < layers.size(); i++)
	{
		layers[i]->calculateOutput();
	}
}

void NeuralNet::addLayer(Layer* layer)
{
	unsigned newNumberLayers = layers.size() + 1;
	bool* newConnectionsGraph = (bool*)MemoryManagement::malloc(
			sizeof(bool) * newNumberLayers * newNumberLayers);
	for (unsigned i = 0; i < newNumberLayers; i++)
	{
		for (unsigned j = 0; j < newNumberLayers; j++)
		{
			if (i == layers.size() || j == layers.size())
			{
				newConnectionsGraph[(i * newNumberLayers) + j] = false;
			}
			else
			{
				newConnectionsGraph[(i * newNumberLayers) + j]
						= connectionsGraph[(i * layers.size()) + j];
			}
		}
	}
	if (connectionsGraph)
	{
		MemoryManagement::free(connectionsGraph);
	}
	connectionsGraph = newConnectionsGraph;

	layers.push_back(layer);
}

void NeuralNet::addLayer(unsigned size, BufferType destinationType,
		FunctionType functiontype)
{
	addLayer(new Layer(size, destinationType, functiontype, getImplementationType()));
}

void NeuralNet::addInputLayer(unsigned size, BufferType bufferType)
{
	//TODO quitar esto
	inputs.push_back(layers.size());
	addLayer(new InputLayer(size, bufferType, getImplementationType()));
}

void NeuralNet::addInputLayer(Interface* interface)
{
	inputs.push_back(layers.size());
	addLayer(new InputLayer(interface, getImplementationType()));
}

void NeuralNet::updateInput(unsigned inputPos, Interface* input)
{
	if (inputPos > inputs.size())
	{
		std::string error = "Cannot get the Input in position " + to_string(inputPos) +
				": there are just " + to_string(inputs.size()) + " Inputs.";
		throw error;
	}
	return ((InputLayer*)(layers[inputs[inputPos]]))->getInputInterface()->copyFromFast(input);
}

unsigned NeuralNet::getNumInputs()
{
	return inputs.size();
}

unsigned char NeuralNet::isInputLayer(unsigned layerPos)
{
	if (layerPos >= layers.size())
	{
		std::string error = "Cannot access the Layer in position " + to_string(layerPos) +
				": there are just " + to_string(layers.size()) + " layers.";
		throw error;
	}
	for (unsigned i = 0; i < inputs.size(); ++i)
	{
		if (inputs[i] == layerPos)
		{
			return 1;
		}
	}
	return 0;
}

Interface* NeuralNet::getOutput(unsigned layerPos)
{
	if (layerPos >= layers.size())
	{
		std::string error = "Cannot access the output in position " + to_string(layerPos) +
				": there are just " + to_string(layers.size()) + " layers.";
		throw error;
	}
	return layers[ layerPos ]->getOutputInterface();
}

unsigned NeuralNet::getNumLayers()
{
	return layers.size();
}

void NeuralNet::randomWeighs(float range)
{
	for (unsigned i = 0; i < layers.size(); i++)
	{
		layers[i]->randomWeighs(range);
	}
}

void NeuralNet::addLayersConnection(unsigned sourceLayerPos, unsigned destinationLayerPos)
{
	if (sourceLayerPos >= layers.size() || destinationLayerPos >= layers.size())
	{
		std::string error = "Cannot connect Layer in position " + to_string(sourceLayerPos) +
				" with Layer in position " + to_string(destinationLayerPos) +
				": there are just " + to_string(layers.size()) + " Layers.";
		throw error;
	}

	layers[destinationLayerPos]->addInput(layers[sourceLayerPos]->getOutput());
	connectionsGraph[(sourceLayerPos * layers.size()) + destinationLayerPos] = true;
}

void NeuralNet::createFeedForwardNet(unsigned inputSize, BufferType inputType,
		unsigned numLayers, unsigned sizeLayers, BufferType hiddenLayersType,
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
	if(layers.size() > 0){
		return layers[0]->getImplementationType();
	} else {
		return implementationType;
	}
}

void NeuralNet::createFullyConnectedNet(unsigned inputSize,
		BufferType inputType, unsigned numLayers, unsigned sizeLayers,
		BufferType hiddenLayersType, FunctionType functiontype)
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
	unsigned inputsSize = inputs.size();
	unsigned layersSize = layers.size();
	fwrite(&inputsSize, sizeof(unsigned), 1, stream);
	fwrite(&layersSize, sizeof(unsigned), 1, stream);

	for (unsigned i = 0; i < layers.size(); i++)
	{
		layers[i]->save(stream);
	}

	fwrite(&inputs[0], sizeof(unsigned) * inputs.size(), 1, stream);
	fwrite(connectionsGraph, sizeof(bool) * layers.size()
			* layers.size(), 1, stream);

	for (unsigned i = 0; i < layers.size(); i++)
	{
		layers[i]->saveWeighs(stream);
	}
}

void NeuralNet::load(FILE* stream)
{
	unsigned inputsSize, layersSize;
	fread(&inputsSize, sizeof(unsigned), 1, stream);
	fread(&layersSize, sizeof(unsigned), 1, stream);

	for (unsigned i = 0; i < layersSize; i++)
	{
		layers.push_back( new Layer(stream, getImplementationType()) );
	}

	inputs.resize(inputsSize);
	fread(&inputs[0], sizeof(unsigned) * inputs.size(), 1, stream);

	size_t graphSize = sizeof(bool) * layers.size() * layers.size();
	connectionsGraph = (bool*)(MemoryManagement::malloc(graphSize));
	fread(connectionsGraph, graphSize, 1, stream);

	stablishConnections();

	for (unsigned i = 0; i < layers.size(); i++)
	{
		layers[i]->loadWeighs(stream);
	}
}

void NeuralNet::stablishConnections()
{
	for (unsigned i = 0; i < inputs.size(); i++)
	{
		unsigned layerPos = inputs[i];
		Buffer* output = layers[layerPos]->getOutput();
		Layer* inputLayer = new InputLayer(output->getSize(),
				output->getBufferType(), getImplementationType());
		inputLayer->getOutput()->copyFrom(output);
		delete (layers[layerPos]);
		layers[layerPos] = inputLayer;
	}

	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < layers.size(); j++)
		{
			if (connectionsGraph[(i * layers.size()) + j])
			{
				addLayersConnection(i, j);
			}
		}
	}
}
