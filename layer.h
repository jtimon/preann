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

	virtual void setSizes(unsigned totalWeighsPerOutput, unsigned ouputSize);
	unsigned weighToPos(unsigned neuronPos, unsigned inputVector, unsigned inputPos);
public:
	Vector* getOutput();

	Vector* getInput(unsigned pos);
	void setSize(unsigned size);
	void resetSize();
	void addInput(Vector* input);
	unsigned getNumberInputs();

	void randomWeighs(float range);
	void save(FILE* stream);
	void load(FILE* stream);
	Layer* newCopy();
	void copyWeighs(Layer* other);

	virtual void calculateOutput();
	virtual Vector* newVector(unsigned size, VectorType vectorType);

	Layer();
	Layer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~Layer();

	void mutateWeigh(float mutationRange);
	void mutateWeighs(float probability, float mutationRange);

	Layer* uniformCrossoverWeighs(Layer* other, float probability);
	Layer* uniformCrossoverNeurons(Layer* other, float probability);

	float getFloatWeigh(unsigned pos);
	void setFloatWeigh(float value, unsigned pos);
	unsigned char getByteWeigh(unsigned pos);
	void setByteWeigh(unsigned char value, unsigned pos);
	float getThreshold(unsigned neuronPos);
	void setThreshold(float value, unsigned  neuronPos);

	void* getThresholdsPtr();
	void* getWeighsPtr();
};

#endif /*ABSTRACTLAYER_H_*/
