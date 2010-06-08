#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "factory.h"

class NeuralNet
{
protected:
	ImplementationType implementationType;

	Layer** layers;
	unsigned char* layerConnectionsGraph;
	unsigned numberLayers;

	Vector** inputs;
	Interface** inputInterfaces;
	unsigned char* inputsToLayersGraph;
	unsigned numberInputs;

	Interface** outputs;
	unsigned* outputLayers;
	unsigned numberOutputs;

	void addLayer(Layer* layer);
	void increaseMaxInputs();
	void increaseMaxLayers();
	void increaseMaxOuputs();
	unsigned getPosInGraph(unsigned source, unsigned destination);
	Layer* getLayer(unsigned pos);
public:

	NeuralNet(ImplementationType implementationType = C);
	virtual ~NeuralNet();

	Interface* createInput(unsigned size, VectorType vectorType);
	Interface* getInput(unsigned pos);
	unsigned getNumInputs();

	Interface* createOutput(unsigned layerPos);
	Interface* getOutput(unsigned outputPos);
	unsigned getNumOutputs();

	void addLayer(unsigned size, VectorType destinationType = FLOAT, FunctionType functiontype = IDENTITY);

	void addInputConnection(unsigned sourceInputPos, unsigned destinationLayerPos);
	void addLayersConnection(unsigned sourceLayerPos, unsigned destinationLayerPos);

	void randomWeighs(float range);
	void save(FILE* stream);
	void load(FILE* stream);
	void resetConnections();

	void createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype = IDENTITY);
	void createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype = IDENTITY);

	virtual void calculateOutput();
};

#endif /*NEURALNET_H_*/
