#ifndef CUDANEURALNET_H_
#define CUDANEURALNET_H_

#include "neuralNet.h"
#include "cudaLayer.h"

extern "C" void** InputsToDevice(void** host_inputs, unsigned* host_inputSizes, VectorType* host_types, unsigned numberInputs);
extern "C" void RefreshDeviceInputs(void** dev_inputs, void** host_inputs, unsigned* host_inputSizes, VectorType* host_types, unsigned numberInputs);
extern "C" void FreeInputs(void** dev_inputs, unsigned numberInputs);

class CudaNeuralNet: public NeuralNet {
protected:

	void** dev_inputs;

	void** host_inputs;
	unsigned* host_inputSizes;
	VectorType* host_types;

	//TODO hay que pensar como se va a gestionar la memoria del device con los experimentos
	char inDevice;
public:
	CudaNeuralNet();
	virtual ~CudaNeuralNet();

	virtual void calculateOutput();

	void hostToDevice();
	void freeDevice();

	//virtual Layer* newLayer();
	//virtual Layer* newLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
};

#endif /* CUDANEURALNET_H_ */
