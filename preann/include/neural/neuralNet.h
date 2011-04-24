#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "factory.h"
#include "inputLayer.h"

class NeuralNet {
	void loadGraphs(FILE* stream);
	void stablishConnections();
protected:
	ImplementationType implementationType;

	Layer** layers;
	unsigned char* layerConnectionsGraph;
	unsigned numberLayers;

	unsigned* inputLayers;
	unsigned numberInputs;

	void addLayer(Layer* layer);
	void increaseMaxInputs();
	void increaseMaxLayers();
	unsigned getPosInGraph(unsigned source, unsigned destination);
	Layer* getLayer(unsigned pos);
	ImplementationType getImplementationType();
public:

	NeuralNet(ImplementationType implementationType = C);
	virtual ~NeuralNet();

	void addInputLayer(unsigned size, VectorType vectorType);
	Interface* getInput(unsigned inputPos);
	unsigned char isInputLayer(unsigned layerPos);
	unsigned getNumInputs();

	Interface* getOutput(unsigned layerPos);

	void addLayer(unsigned size, VectorType destinationType = FLOAT,
			FunctionType functiontype = IDENTITY);
	unsigned getNumLayers();

	void addLayersConnection(unsigned sourceLayerPos,
			unsigned destinationLayerPos);

	virtual void calculateOutput();
	void randomWeighs(float range);
	void save(FILE* stream);
	void load(FILE* stream);

	void createFeedForwardNet(unsigned inputSize, VectorType inputType,
			unsigned numLayers, unsigned sizeLayers,
			VectorType hiddenLayersType, FunctionType functiontype = IDENTITY);
	void createFullyConnectedNet(unsigned inputSize, VectorType inputType,
			unsigned numLayers, unsigned sizeLayers,
			VectorType hiddenLayersType, FunctionType functiontype = IDENTITY);

};

#endif /*NEURALNET_H_*/
