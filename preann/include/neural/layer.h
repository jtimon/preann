#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "connection.h"
#include "factory.h"

class Layer {
protected:
	Layer() {};
	Connection** connections;
	unsigned numberInputs;

	Connection* thresholds;
	Vector* output;
	Interface* tOuputInterface;
	FunctionType functionType;
	ImplementationType getImplementationType();
	Vector* newVector(FILE* stream);
	Vector* newVector(unsigned size, VectorType vectorType);
public:
	Layer(unsigned size, VectorType outputType, FunctionType functionType, ImplementationType implementationType);
	Layer(FILE* stream, ImplementationType implementationType);
	virtual ~Layer();

	void addInput(Vector* input);
	void calculateOutput();

	void randomWeighs(float range);
	void copyWeighs(Layer* sourceLayer);
	void loadWeighs(FILE* stream);
	void saveWeighs(FILE* stream);
	void save(FILE* stream);

	unsigned getNumberInputs();
	Vector* getInput(unsigned pos);
	Connection* getConnection(unsigned inputPos);
	Vector* getOutput();
	Interface* getOutputInterface();
	Connection* getThresholds();
	FunctionType getFunctionType();
};

#endif /*ABSTRACTLAYER_H_*/
