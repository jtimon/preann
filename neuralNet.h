#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "factory.h"

class NeuralNet
{
protected:

	Layer** layers;
	unsigned char* layerConnectionsGraph;
	unsigned numberLayers;

	Vector** inputs;
	unsigned char* inputsToLayersGraph;
	unsigned numberInputs;

	Vector** outputs;
	unsigned* outputLayers;
	unsigned numberOutputs;

	void addLayer(Layer* layer);
	void increaseMaxInputs();
	void increaseMaxLayers();
	void increaseMaxOuputs();
	unsigned getPosInGraph(unsigned source, unsigned destination);
public:
	ImplementationType implementationType;
	NeuralNet();
	NeuralNet(ImplementationType implementationType);
	virtual ~NeuralNet();

	void addInput(Vector* input);
	void setInput(Vector* input);
	void addOutput(unsigned layerPos);
	void addLayer(unsigned  size, VectorType sourceType, VectorType destinationType);
	void addLayer(unsigned  size, VectorType sourceType, VectorType destinationType, FunctionType functiontype);
	void addInputConnection(unsigned sourceInputPos, unsigned destinationLayerPos);
	void addLayersConnection(unsigned sourceLayerPos, unsigned destinationLayerPos);

	Vector* getOutput(unsigned outputPos);
	unsigned getNumOutputs();
	void randomWeighs(float range);
	void save(FILE* stream);
	void load(FILE* stream);
	void resetConnections();

	void createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType);
	void createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype);
	void createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype, unsigned floatOutputSize, unsigned bitOutputSize);
	void createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType);
	void createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype);

	virtual void calculateOutput();

	virtual Layer* newLayer();
	virtual Layer* newLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual Vector* newVector(unsigned size, VectorType vectorType);
};

#endif /*NEURALNET_H_*/
