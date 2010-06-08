#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "vector.h"

typedef enum {C, SSE2, CUDA, CUDA2} ImplementationType;

class Layer
{
protected:

	Vector** inputs;
	void** weighs;
	unsigned numberInputs;

	float* thresholds;
	Vector* output;

	FunctionType functionType;

	virtual float* negativeThresholds() = 0;
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results) = 0;

	virtual void* newWeighs(unsigned inputSize, VectorType inputType) = 0;
	virtual void saveWeighs(FILE* stream) = 0;
	virtual void loadWeighs(FILE* stream) = 0;

	virtual void mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation) = 0;
	virtual void mutateThreshold(unsigned outputPos, float mutation) = 0;

	Layer();
public:
	virtual ~Layer();
	virtual ImplementationType getImplementationType() = 0;

	virtual void copyWeighs(Layer* sourceLayer) = 0;
	virtual void randomWeighs(float range) = 0;
	virtual void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector) = 0;

	void checkCompatibility(Layer* layer);
	void calculateOutput();
	void addInput(Vector* input);
	void save(FILE* stream);
	void load(FILE* stream);

	void swapWeighs(Layer* layer);
	unsigned getNumberInputs();
	Vector* getInput(unsigned pos);
	Vector* getOutput();
	float* getThresholdsPtr();
	void* getWeighsPtr(unsigned inputPos);

	void mutateWeigh(float mutationRange);
	void mutateWeighs(float probability, float mutationRange);
	void crossoverNeurons(Layer* other, Interface* bitVector);
	void crossoverInput(Layer* other, unsigned inputLayer, Interface* bitVector);

};

#endif /*ABSTRACTLAYER_H_*/
