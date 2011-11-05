#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "factory.h"
#include "inputLayer.h"

class NeuralNet {
	void loadGraphs(FILE* stream);
	void stablishConnections();
protected:
	ImplementationType implementationType;

	vector<InputLayer*> inputs;
	vector<Layer*> layers;

	SimpleGraph inputConnectionsGraph;
	SimpleGraph connectionsGraph;

	Layer* getLayer(unsigned pos);
	ImplementationType getImplementationType();
public:

	NeuralNet(ImplementationType implementationType = IT_C);
	virtual ~NeuralNet();

	void addInputLayer(Interface* interface);
	void addInputLayer(unsigned size, BufferType bufferType);
	void updateInput(unsigned inputPos, Interface* input);
	unsigned getNumInputs();

	Interface* getOutput(unsigned layerPos);

	void addLayer(unsigned size, BufferType destinationType = BT_FLOAT,
			FunctionType functiontype = FT_IDENTITY);
	unsigned getNumLayers();

	void addInputConnection(unsigned sourceInputPos, unsigned destinationLayerPos);
	void addLayersConnection(unsigned sourceLayerPos, unsigned destinationLayerPos);

	virtual void calculateOutput();
	void randomWeighs(float range);
	void save(FILE* stream);
	void load(FILE* stream);

	void createFeedForwardNet(unsigned inputSize, BufferType inputType,
			unsigned numLayers, unsigned sizeLayers,
			BufferType hiddenLayersType, FunctionType functiontype = FT_IDENTITY);
	void createFullyConnectedNet(unsigned inputSize, BufferType inputType,
			unsigned numLayers, unsigned sizeLayers,
			BufferType hiddenLayersType, FunctionType functiontype = FT_IDENTITY);

};

#endif /*NEURALNET_H_*/
