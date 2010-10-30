#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "vector.h"

class Layer
{
private:
	Layer() {};
protected:
	Vector** inputs;
	Vector** connections;
	unsigned numberInputs;

	Vector* thresholds;
	Vector* output;

	FunctionType functionType;


	ImplementationType getImplementationType() {
		return output->getImplementationType();
	};

	void mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation);
	void mutateThreshold(unsigned outputPos, float mutation);

	Vector* newVector(FILE* stream);
	Vector* newVector(unsigned size, VectorType vectorType);
public:
	Layer(unsigned size, VectorType outputType, FunctionType functionType, ImplementationType implementationType);
	Layer(FILE* stream, ImplementationType implementationType);
	virtual ~Layer();

	void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector);

	void save(FILE* stream);
	void checkCompatibility(Layer* layer);
	void calculateOutput();
	void addInput(Vector* input);
	void setInput(Vector* input, unsigned pos);

	void copyWeighs(Layer* sourceLayer);
	void randomWeighs(float range);

	void swapWeighs(Layer* layer);
	unsigned getNumberInputs();
	Vector* getInput(unsigned pos);
	Vector* getOutput();
	float* getThresholdsPtr();
	Vector* getConnection(unsigned inputPos);
	FunctionType getFunctionType();

	void mutateWeigh(float mutationRange);
	void mutateWeighs(float probability, float mutationRange);
	void crossoverNeurons(Layer* other, Interface* bitVector);
	void crossoverInput(Layer* other, unsigned inputLayer, Interface* bitVector);

};

#endif /*ABSTRACTLAYER_H_*/
