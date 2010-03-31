#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "vector.h"

class Layer
{
protected:

	Vector** inputs;
	void** weighs;
	unsigned numberInputs;

	float* thresholds;
	Vector* output;

	VectorType outputType;
	FunctionType functionType;

	virtual void saveWeighs(FILE* stream) = 0;
	virtual void loadWeighs(FILE* stream) = 0;
	virtual float* negativeThresholds() = 0;
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results) = 0;
	virtual void* newWeighs(unsigned inputSize, VectorType inputType) = 0;

// To allow GA trainer work:
	/*float getFloatWeigh(unsigned pos);
	void setFloatWeigh(float value, unsigned pos);
	unsigned char getByteWeigh(unsigned pos);
	void setByteWeigh(unsigned char value, unsigned pos);
	float getThreshold(unsigned neuronPos);
	void setThreshold(float value, unsigned  neuronPos);
	void* getThresholdsPtr();
	void* getWeighsPtr();*/
public:
	Layer(VectorType outputType, FunctionType functionType);
	virtual ~Layer();

	virtual void randomWeighs(float range) = 0;

	void calculateOutput();
	void addInput(Vector* input);
	void save(FILE* stream);
	void load(FILE* stream);

	Vector* getOutput();

// To allow GA trainer work:
	virtual Layer* newCopy() = 0;
	/*void copyWeighs(Layer* other);

	void mutateWeigh(float mutationRange);
	void mutateWeighs(float probability, float mutationRange);

	Layer** crossoverNeurons(Layer* other, Interface* bitVector);
	Layer** crossoverWeighs(Layer* other, Interface* bitVector);

	unsigned getNumberNeurons();
	unsigned getNumberWeighs();*/
};

#endif /*ABSTRACTLAYER_H_*/
