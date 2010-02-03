#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "cudaLayer.h"
#include "xmmLayer.h"

class NeuralNet
{
protected:

	Layer** layers;
	unsigned char* layerConnectionsGraph;
	unsigned numberLayers;
	unsigned maxLayers;

	Vector** inputs;
	unsigned char* inputsToLayersGraph;
	unsigned numberInputs;
	unsigned maxInputs;

	Vector** outputs;
	int* outputLayers;
	unsigned numberOutputs;
	unsigned maxOutputs;

	void addLayer(Layer* layer);
	void changeMaxInputs(unsigned newMaxInputs);
	void changeMaxLayers(unsigned newMaxLayers);
	void changeMaxOuputs(unsigned newMaxOutputs);
	unsigned getPosInGraph(unsigned source, unsigned destination);
public:
	NeuralNet();
	NeuralNet(unsigned maxInputs, unsigned maxLayers, unsigned maxOutputs);
	virtual ~NeuralNet();
	void addInput(Vector* input);
	Vector* getOutput(unsigned outputPos);
	unsigned getNumOutputs();
	void randomWeighs(float range);
	void addLayer(unsigned  size, VectorType sourceType, VectorType destinationType);
	void addInputConnection(unsigned sourceInputPos, unsigned destinationLayerPos);
	void addLayersConnection(unsigned sourceLayerPos, unsigned destinationLayerPos);
	void setLayerAsOutput(unsigned layerPos);
	void createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType);
	void createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype);
	void createFeedForwardNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype, unsigned floatOutputSize, unsigned bitOutputSize);
	void createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType);
	void createFullyConnectedNet(unsigned numLayers, unsigned sizeLayers, VectorType hiddenLayersType, FunctionType functiontype);

	virtual void calculateOutput();
	void addLayer(unsigned  size, VectorType sourceType, VectorType destinationType, FunctionType functiontype);

	void save(FILE* stream);
	void load(FILE* stream);

	virtual Layer* newLayer();
	virtual Layer* newLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual Vector* newVector(unsigned size, VectorType vectorType);
	virtual void freeNeuralNet();
};

#endif /*NEURALNET_H_*/
