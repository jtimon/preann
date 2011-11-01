#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "connection.h"
#include "factory.h"

class Layer {
protected:
	Layer() {};
	std::vector<Connection*> connections;
//	Connection** connections;
//	unsigned numberInputs;

	Connection* thresholds;
	Buffer* output;
	Interface* tOuputInterface;
	FunctionType functionType;
	Buffer* newBuffer(FILE* stream);
	Buffer* newBuffer(unsigned size, BufferType bufferType);
public:
	Layer(unsigned size, BufferType outputType, FunctionType functionType, ImplementationType implementationType);
	Layer(FILE* stream, ImplementationType implementationType);
	virtual ~Layer();

	virtual void addInput(Buffer* input);
	virtual void calculateOutput();

	virtual void randomWeighs(float range);
	virtual void copyWeighs(Layer* sourceLayer);

	void loadWeighs(FILE* stream);
	void saveWeighs(FILE* stream);
	virtual void save(FILE* stream);

	unsigned getNumberInputs();
	Buffer* getInput(unsigned pos);
	Connection* getConnection(unsigned inputPos);
	Buffer* getOutput();
	Interface* getOutputInterface();
	Connection* getThresholds();
	FunctionType getFunctionType();
	ImplementationType getImplementationType();
};

#endif /*ABSTRACTLAYER_H_*/
