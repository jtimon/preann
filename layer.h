#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "vector.h"

class Layer
{
protected:

	Vector** inputs;
	unsigned numberInputs;
	unsigned totalWeighsPerOutput;

	void* weighs;

	float* thresholds;

	Vector* output;

	VectorType inputType;
	VectorType outputType;
	FunctionType functionType;

	virtual void saveWeighs(FILE* stream) = 0;
	virtual void loadWeighs(FILE* stream) = 0;

// To allow GA trainer work:
	float getFloatWeigh(unsigned pos);
	void setFloatWeigh(float value, unsigned pos);
	unsigned char getByteWeigh(unsigned pos);
	void setByteWeigh(unsigned char value, unsigned pos);
	float getThreshold(unsigned neuronPos);
	void setThreshold(float value, unsigned  neuronPos);
	void* getThresholdsPtr();
	void* getWeighsPtr();
public:
	virtual void setSizes(unsigned totalWeighsPerOutput, unsigned ouputSize) = 0;
	virtual void calculateOutput() = 0;
	virtual void randomWeighs(float range);

	Vector* getOutput();
	Vector* getInput(unsigned pos);
	void setSize(unsigned size);
	void resetSize();
	void addInput(Vector* input);
	unsigned getNumberInputs();

	void save(FILE* stream);
	void load(FILE* stream);



	Layer();
	Layer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~Layer();

// To allow GA trainer work:
	virtual Layer* newCopy() = 0;
	void copyWeighs(Layer* other);

	void mutateWeigh(float mutationRange);
	void mutateWeighs(float probability, float mutationRange);

	Layer** crossoverNeurons(Layer* other, Interface* bitVector);
	Layer** crossoverWeighs(Layer* other, Interface* bitVector);

	unsigned getNumberNeurons();
	unsigned getNumberWeighs();
};

#endif /*ABSTRACTLAYER_H_*/
