#ifndef CUDALAYER_H_
#define CUDALAYER_H_

#include "layer.h"
#include "cuda_code.h"

extern "C" void LayerCalculation(struct_Layer* d_layer, unsigned threads, VectorType inputType, VectorType outputType);
extern "C" void LayerCalculation2(struct_Layer* d_layer, unsigned block_size, VectorType inputType, VectorType outputType);
extern "C" void LayerCalculation3(struct_Layer* d_layer, unsigned block_size, VectorType inputType, VectorType outputType);

extern "C" struct_Layer* LayerHostToDevice(struct_Layer* h_layer, VectorType inputType, VectorType outputType);
extern "C" void SetInputsInDevice(struct_Layer* d_layer, void** inputs);
extern "C" void OutputToHost(void* output, struct_Layer* d_layer, VectorType outputType);
extern "C" void FreeDevice(struct_Layer* d_layer);

class CudaLayer : public Layer
{
protected:

	struct_Layer* deviceLayer;

	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);
public:
	static unsigned block_size;
	static unsigned version;
	CudaLayer();
	CudaLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~CudaLayer();

	//New methods
	void toDevice();
	void setInputsInDevice(void** inputs);
	float* getDeviceOutput();
	void freeDevice();
	void outputToHost();

	//redefined methods
	virtual void calculateOutput();
	virtual Layer* newCopy();
	virtual void setSizes(unsigned totalWeighsPerOutput, unsigned ouputSize);

	
};

#endif /*CUDALAYER_H_*/
