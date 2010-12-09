#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "factory.h"
#include "inputLayer.h"
#include "outputLayer.h"

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

	void addInputLayer(unsigned size, VectorType vectorType);
	Interface* getInput(unsigned inputPos);
	unsigned char isInputLayer(unsigned layerPos);
	unsigned getNumInputs();

	void addOutputLayer(unsigned size, VectorType destinationType, FunctionType functiontype);
	Interface* getOutput(unsigned outputPos);
	unsigned char isOutputLayer(unsigned layerPos);
	unsigned getNumOutputs();

	void addLayer(unsigned size, VectorType destinationType = FLOAT,
			FunctionType functiontype = IDENTITY);

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
