#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "vector.h"

typedef enum {C, SSE2, CUDA, CUDA2} ImplementationType;

class Layer
{
protected:

	Vector** inputs;
	Vector** connections;
	unsigned numberInputs;

	Vector* thresholds;
	Vector* output;

	FunctionType functionType;

	virtual float* negativeThresholds() = 0;
	//TODO change parameter results to a Vector
	virtual void inputCalculation(Vector* input, Vector* inputWeighs, Vector* results) = 0;

	void mutateWeigh(unsigned outputPos, unsigned inputLayer, unsigned inputPos, float mutation);
	void mutateThreshold(unsigned outputPos, float mutation);

	Layer();
	Vector* newVector(FILE* stream);
	Vector* newVector(unsigned size, VectorType vectorType);
public:
	virtual void init(unsigned size, VectorType outputType, FunctionType functionType);
	virtual ~Layer();
	virtual ImplementationType getImplementationType() = 0;

	virtual void copyWeighs(Layer* sourceLayer) = 0;
	virtual void crossoverWeighs(Layer* other, unsigned inputLayer, Interface* bitVector) = 0;


	void checkCompatibility(Layer* layer);
	void calculateOutput();
	void addInput(Vector* input);
	void setInput(Vector* input, unsigned pos);
	virtual void save(FILE* stream);
	virtual void load(FILE* stream);
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
